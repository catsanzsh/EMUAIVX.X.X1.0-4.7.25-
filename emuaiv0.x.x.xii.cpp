#include <QApplication>
#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QLabel>
#include <QStatusBar>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QThread>
#include <QOpenGLWidget>
#include <QTimer>
#include <QPixmap>
#include <QPainter>
#include <atomic>
#include <iostream>
#include <fstream>
#include <unordered_map>
#include <functional>
#include <cstring>
#include <vector>
#include <mutex>
#include <condition_variable>

// ========== Enhanced Constants and Basic Definitions ========== //

#define MEMORY_SIZE      0x800000    // 8 MB typical RDRAM
#define R4300_REGS       32
#define DISPLAY_WIDTH    640
#define DISPLAY_HEIGHT   480
#define CACHE_SIZE       8192        // Size of recompiled block cache entries

// ROM header structure
struct ROMHeader {
    uint8_t  init_PI_BSB_DOM1_LAT_REG;
    uint8_t  init_PI_BSB_DOM1_PGS_REG;
    uint8_t  init_PI_BSB_DOM1_PWD_REG;
    uint8_t  init_PI_BSB_DOM1_PGS_REG2;
    uint32_t clockRate;
    uint32_t pc;                 // Program counter at boot
    uint32_t release;
    uint32_t crc1;
    uint32_t crc2;
    uint64_t unknown0;
    char     name[20];           // ROM name
    uint32_t unknown1;
    uint32_t manufacturer_ID;
    uint16_t cartridge_ID;
    char     country_code[2];
};

// Enhanced CPU structure with more registers and status
struct CPU {
    uint32_t pc = 0;             // Program counter
    uint32_t regs[R4300_REGS] = {0};  // General purpose registers
    uint32_t hi = 0;             // Multiply/divide high result
    uint32_t lo = 0;             // Multiply/divide low result
    bool     running = false;    // CPU running state
    bool     delaySlot = false;  // Currently in a branch delay slot
    uint32_t nextPC = 0;         // PC after delay slot
    uint32_t cycles = 0;         // Cycle counter
    
    // Debug/stats
    uint64_t instructions = 0;   // Total instructions executed
    uint64_t dynarecHits = 0;    // Number of dynarec block executions
    uint64_t interpreterInst = 0; // Number of interpreted instructions
    
    // Reset the CPU state
    void reset() {
        pc = 0;
        std::memset(regs, 0, sizeof(regs));
        hi = lo = 0;
        running = false;
        delaySlot = false;
        nextPC = 0;
        cycles = 0;
        instructions = 0;
        dynarecHits = 0;
        interpreterInst = 0;
    }
};

// Global CPU/memory
static CPU cpu;
static uint8_t* gMem = nullptr;
static ROMHeader* gROMHeader = nullptr;

static std::atomic<bool> gEmulationRunning{false};
static std::atomic<bool> gStopRequested{false};
static std::atomic<bool> gPaused{false};

// ========== Enhanced DynaRec System ========== //
class Dynarec {
public:
    // Function pointer type for a block of recompiled code
    using RecompiledBlockFunc = void(*)();
    
    Dynarec() {
        // Reserve space in the cache map for better performance
        blockCache.reserve(CACHE_SIZE);
    }
    
    // We'll map an address (PC) -> a function pointer that executes that block
    std::unordered_map<uint32_t, RecompiledBlockFunc> blockCache;
    
    // Stats
    size_t cacheHits = 0;
    size_t cacheMisses = 0;
    
    // Recompile a block starting at `pc`, return a function pointer
    // This is extremely simplified. A real dynarec would decode many instructions,
    // produce x86 or x64 machine code, then store it in a code buffer.
    RecompiledBlockFunc recompileBlock(uint32_t pc) {
        // Check if we already have it
        auto it = blockCache.find(pc);
        if (it != blockCache.end()) {
            cacheHits++;
            return it->second;
        }
        
        cacheMisses++;
        
        // Find block end (branch instruction or maximum block size)
        const int MAX_BLOCK_SIZE = 16; // Maximum instructions per block
        uint32_t endPC = pc;
        int blockSize = 0;
        bool endsWithBranch = false;
        
        // Analyze the block to find its end (first branch or maximum size)
        for (int i = 0; i < MAX_BLOCK_SIZE; i++) {
            if (endPC + 3 >= MEMORY_SIZE) break;
            
            uint32_t instr = (gMem[endPC] << 24) | (gMem[endPC+1] << 16) |
                             (gMem[endPC+2] << 8) | gMem[endPC+3];
            uint8_t opcode = (instr >> 26) & 0x3F;
            
            // Check if it's a branch instruction
            if (opcode == 0x02 || opcode == 0x03 || // J, JAL
                opcode == 0x04 || opcode == 0x05 || // BEQ, BNE
                opcode == 0x06 || opcode == 0x07 || // BLEZ, BGTZ
                (opcode == 0x00 && ((instr & 0x3F) == 0x08 || (instr & 0x3F) == 0x09))) { // JR, JALR
                endsWithBranch = true;
                endPC += 4; // Include the branch instruction
                blockSize++;
                break;
            }
            
            endPC += 4;
            blockSize++;
        }
        
        // Create a function that will interpret all instructions in this block
        auto blockStartPC = pc;
        auto blockEndPC = endPC;
        auto generatedFunc = [blockStartPC, blockEndPC]() {
            // Execute all instructions in this block
            uint32_t currentPC = blockStartPC;
            while (currentPC < blockEndPC && cpu.running && !gPaused.load()) {
                // Interpret a single instruction
                interpretSingleInstruction(currentPC);
                
                // If we hit a branch, we might need to stop executing the block
                if (cpu.delaySlot) {
                    // Execute one more instruction (delay slot)
                    currentPC += 4;
                    if (currentPC < blockEndPC) {
                        interpretSingleInstruction(currentPC);
                    }
                    cpu.delaySlot = false;
                    cpu.pc = cpu.nextPC;
                    break;
                }
                currentPC += 4;
            }
        };
        
        // Convert lambda to function pointer (in a real dynarec, this would be JIT compilation)
        RecompiledBlockFunc fptr = createTrampoline(generatedFunc);
        
        // Cache the result (limit cache size to prevent memory issues)
        if (blockCache.size() >= CACHE_SIZE) {
            // Simple strategy: just clear the entire cache when it gets too big
            blockCache.clear();
        }
        
        blockCache[pc] = fptr;
        return fptr;
    }
    
    // Called by CPU loop to get or generate code
    inline void runBlock(uint32_t pc) {
        RecompiledBlockFunc block = recompileBlock(pc);
        // Jump to generated code:
        block();
        // Count dynarec hit
        cpu.dynarecHits++;
    }
    
    // Clear the dynarec cache (e.g., when loading a new ROM)
    void clearCache() {
        blockCache.clear();
        cacheHits = 0;
        cacheMisses = 0;
    }
    
private:
    // Stub function that converts a C++ lambda to a function pointer.
    // Real dynarecs allocate memory, write machine code, etc.
    static RecompiledBlockFunc createTrampoline(std::function<void()> fn) {
        // Store the std::function in a static array and return a function pointer that calls it
        static std::vector<std::function<void()>> s_functions;
        s_functions.push_back(fn);
        size_t index = s_functions.size() - 1;
        
        // Helper to call the stored function
        struct FnHolder {
            static void call(size_t i) { s_functions[i](); }
        };
        
        // Create a trampolines array (simulating compiled blocks)
        static std::vector<void(*)()> s_trampolines;
        while (s_trampolines.size() <= index) {
            s_trampolines.push_back(nullptr);
        }
        
        // Store the trampoline function
        s_trampolines[index] = [index]() { FnHolder::call(index); };
        
        return s_trampolines[index];
    }
    
    // Expanded interpreter for more N64 instructions
    static void interpretSingleInstruction(uint32_t addr) {
        if (!cpu.running || gPaused.load()) return;
        
        // Ensure addr is valid
        if (addr + 3 >= MEMORY_SIZE) {
            cpu.running = false;
            return;
        }
        
        // Fetch the instruction (big endian)
        uint32_t instruction = (gMem[addr] << 24) | (gMem[addr+1] << 16) |
                               (gMem[addr+2] << 8) | gMem[addr+3];
        
        // Increment PC (may be overridden by branch instructions)
        uint32_t oldPC = cpu.pc;
        cpu.pc = addr + 4;
        
        // Count this instruction
        cpu.instructions++;
        cpu.interpreterInst++;
        
        // Decode fields common to R-type instructions
        uint8_t opcode = (instruction >> 26) & 0x3F;
        uint8_t rs = (instruction >> 21) & 0x1F;
        uint8_t rt = (instruction >> 16) & 0x1F;
        uint8_t rd = (instruction >> 11) & 0x1F;
        uint8_t shamt = (instruction >> 6) & 0x1F;
        uint8_t funct = instruction & 0x3F;
        
        // I-type immediate value (sign extended)
        int16_t imm = static_cast<int16_t>(instruction & 0xFFFF);
        
        // Execute instruction based on opcode
        switch(opcode) {
            case 0x00: // SPECIAL
                switch(funct) {
                    case 0x00: // SLL - Shift Left Logical
                        cpu.regs[rd] = cpu.regs[rt] << shamt;
                        break;
                        
                    case 0x02: // SRL - Shift Right Logical
                        cpu.regs[rd] = cpu.regs[rt] >> shamt;
                        break;
                        
                    case 0x03: // SRA - Shift Right Arithmetic
                        cpu.regs[rd] = static_cast<int32_t>(cpu.regs[rt]) >> shamt;
                        break;
                        
                    case 0x08: // JR - Jump Register
                        cpu.delaySlot = true;
                        cpu.nextPC = cpu.regs[rs];
                        break;
                        
                    case 0x09: // JALR - Jump and Link Register
                        cpu.delaySlot = true;
                        cpu.nextPC = cpu.regs[rs];
                        cpu.regs[rd] = addr + 8; // Return address
                        break;
                        
                    case 0x20: // ADD - Add (with overflow)
                        // Note: Real implementation should check for overflow
                        cpu.regs[rd] = cpu.regs[rs] + cpu.regs[rt];
                        break;
                        
                    case 0x21: // ADDU - Add Unsigned (no overflow)
                        cpu.regs[rd] = cpu.regs[rs] + cpu.regs[rt];
                        break;
                        
                    case 0x22: // SUB - Subtract (with overflow)
                        // Note: Real implementation should check for overflow
                        cpu.regs[rd] = cpu.regs[rs] - cpu.regs[rt];
                        break;
                        
                    case 0x23: // SUBU - Subtract Unsigned (no overflow)
                        cpu.regs[rd] = cpu.regs[rs] - cpu.regs[rt];
                        break;
                        
                    case 0x24: // AND - Bitwise AND
                        cpu.regs[rd] = cpu.regs[rs] & cpu.regs[rt];
                        break;
                        
                    case 0x25: // OR - Bitwise OR
                        cpu.regs[rd] = cpu.regs[rs] | cpu.regs[rt];
                        break;
                        
                    case 0x26: // XOR - Bitwise XOR
                        cpu.regs[rd] = cpu.regs[rs] ^ cpu.regs[rt];
                        break;
                        
                    case 0x27: // NOR - Bitwise NOR
                        cpu.regs[rd] = ~(cpu.regs[rs] | cpu.regs[rt]);
                        break;
                        
                    case 0x2A: // SLT - Set Less Than
                        cpu.regs[rd] = (static_cast<int32_t>(cpu.regs[rs]) < 
                                       static_cast<int32_t>(cpu.regs[rt])) ? 1 : 0;
                        break;
                        
                    case 0x2B: // SLTU - Set Less Than Unsigned
                        cpu.regs[rd] = (cpu.regs[rs] < cpu.regs[rt]) ? 1 : 0;
                        break;
                        
                    default:
                        // Unknown SPECIAL instruction
                        std::cout << "Unknown SPECIAL instruction: " << std::hex << funct 
                                  << " at PC " << addr << std::dec << std::endl;
                        break;
                }
                break;
                
            case 0x01: // REGIMM
                switch(rt) {
                    case 0x00: // BLTZ - Branch if Less Than Zero
                        if (static_cast<int32_t>(cpu.regs[rs]) < 0) {
                            cpu.delaySlot = true;
                            cpu.nextPC = addr + 4 + (imm << 2);
                        }
                        break;
                        
                    case 0x01: // BGEZ - Branch if Greater Than or Equal to Zero
                        if (static_cast<int32_t>(cpu.regs[rs]) >= 0) {
                            cpu.delaySlot = true;
                            cpu.nextPC = addr + 4 + (imm << 2);
                        }
                        break;
                        
                    default:
                        // Unknown REGIMM instruction
                        std::cout << "Unknown REGIMM instruction: " << std::hex << rt 
                                  << " at PC " << addr << std::dec << std::endl;
                        break;
                }
                break;
                
            case 0x02: // J - Jump
                cpu.delaySlot = true;
                cpu.nextPC = (addr & 0xF0000000) | ((instruction & 0x03FFFFFF) << 2);
                break;
                
            case 0x03: // JAL - Jump And Link
                cpu.delaySlot = true;
                cpu.nextPC = (addr & 0xF0000000) | ((instruction & 0x03FFFFFF) << 2);
                cpu.regs[31] = addr + 8; // Return address in r31
                break;
                
            case 0x04: // BEQ - Branch if Equal
                if (cpu.regs[rs] == cpu.regs[rt]) {
                    cpu.delaySlot = true;
                    cpu.nextPC = addr + 4 + (imm << 2);
                }
                break;
                
            case 0x05: // BNE - Branch if Not Equal
                if (cpu.regs[rs] != cpu.regs[rt]) {
                    cpu.delaySlot = true;
                    cpu.nextPC = addr + 4 + (imm << 2);
                }
                break;
                
            case 0x06: // BLEZ - Branch if Less Than or Equal to Zero
                if (static_cast<int32_t>(cpu.regs[rs]) <= 0) {
                    cpu.delaySlot = true;
                    cpu.nextPC = addr + 4 + (imm << 2);
                }
                break;
                
            case 0x07: // BGTZ - Branch if Greater Than Zero
                if (static_cast<int32_t>(cpu.regs[rs]) > 0) {
                    cpu.delaySlot = true;
                    cpu.nextPC = addr + 4 + (imm << 2);
                }
                break;
                
            case 0x08: // ADDI - Add Immediate (with overflow)
                // Note: Real implementation should check for overflow
                cpu.regs[rt] = cpu.regs[rs] + imm;
                break;
                
            case 0x09: // ADDIU - Add Immediate Unsigned (no overflow)
                cpu.regs[rt] = cpu.regs[rs] + imm;
                break;
                
            case 0x0A: // SLTI - Set Less Than Immediate
                cpu.regs[rt] = (static_cast<int32_t>(cpu.regs[rs]) < static_cast<int32_t>(imm)) ? 1 : 0;
                break;
                
            case 0x0B: // SLTIU - Set Less Than Immediate Unsigned
                cpu.regs[rt] = (cpu.regs[rs] < static_cast<uint32_t>(imm)) ? 1 : 0;
                break;
                
            case 0x0C: // ANDI - Bitwise AND Immediate
                cpu.regs[rt] = cpu.regs[rs] & (imm & 0xFFFF); // Zero-extended immediate
                break;
                
            case 0x0D: // ORI - Bitwise OR Immediate
                cpu.regs[rt] = cpu.regs[rs] | (imm & 0xFFFF); // Zero-extended immediate
                break;
                
            case 0x0E: // XORI - Bitwise XOR Immediate
                cpu.regs[rt] = cpu.regs[rs] ^ (imm & 0xFFFF); // Zero-extended immediate
                break;
                
            case 0x0F: // LUI - Load Upper Immediate
                cpu.regs[rt] = imm << 16;
                break;
                
            case 0x20: // LB - Load Byte
                {
                    uint32_t addr = cpu.regs[rs] + imm;
                    if (addr < MEMORY_SIZE) {
                        // Sign extend from 8 bits
                        cpu.regs[rt] = static_cast<int8_t>(gMem[addr]);
                    }
                }
                break;
                
            case 0x21: // LH - Load Halfword
                {
                    uint32_t addr = cpu.regs[rs] + imm;
                    if (addr + 1 < MEMORY_SIZE) {
                        // Sign extend from 16 bits
                        cpu.regs[rt] = static_cast<int16_t>((gMem[addr] << 8) | gMem[addr+1]);
                    }
                }
                break;
                
            case 0x23: // LW - Load Word
                {
                    uint32_t addr = cpu.regs[rs] + imm;
                    if (addr + 3 < MEMORY_SIZE) {
                        cpu.regs[rt] = (gMem[addr] << 24) | (gMem[addr+1] << 16) |
                                      (gMem[addr+2] << 8) | gMem[addr+3];
                    }
                }
                break;
                
            case 0x28: // SB - Store Byte
                {
                    uint32_t addr = cpu.regs[rs] + imm;
                    if (addr < MEMORY_SIZE) {
                        gMem[addr] = cpu.regs[rt] & 0xFF;
                    }
                }
                break;
                
            case 0x29: // SH - Store Halfword
                {
                    uint32_t addr = cpu.regs[rs] + imm;
                    if (addr + 1 < MEMORY_SIZE) {
                        gMem[addr] = (cpu.regs[rt] >> 8) & 0xFF;
                        gMem[addr+1] = cpu.regs[rt] & 0xFF;
                    }
                }
                break;
                
            case 0x2B: // SW - Store Word
                {
                    uint32_t addr = cpu.regs[rs] + imm;
                    if (addr + 3 < MEMORY_SIZE) {
                        gMem[addr] = (cpu.regs[rt] >> 24) & 0xFF;
                        gMem[addr+1] = (cpu.regs[rt] >> 16) & 0xFF;
                        gMem[addr+2] = (cpu.regs[rt] >> 8) & 0xFF;
                        gMem[addr+3] = cpu.regs[rt] & 0xFF;
                    }
                }
                break;
                
            default:
                // Unknown opcode
                std::cout << "Unknown opcode: " << std::hex << static_cast<int>(opcode) 
                          << " at PC " << addr << std::dec << std::endl;
                break;
        }
        
        // Register 0 is always 0 in MIPS
        cpu.regs[0] = 0;
        
        // Check if PC is out of bounds
        if (cpu.pc >= MEMORY_SIZE) {
            cpu.running = false;
        }
    }
};

static Dynarec gDynarec;

// ========== Enhanced HLE Video Plugin ========== //
class HLEVideoPlugin {
public:
    HLEVideoPlugin() : frameBuffer(nullptr), width(DISPLAY_WIDTH), height(DISPLAY_HEIGHT) {
        // Allocate framebuffer
        frameBuffer = new uint32_t[width * height];
        clearFramebuffer();
    }
    
    ~HLEVideoPlugin() {
        if (frameBuffer) {
            delete[] frameBuffer;
            frameBuffer = nullptr;
        }
    }
    
    virtual void initialize() {
        std::cout << "[HLEVideo] Initialized (UltraHLE style HLE).\n";
        clearFramebuffer();
    }
    
    virtual void renderFrame() {
        // UltraHLE style high-level rendering
        // In a real emulator, this would detect display list commands and render 3D graphics
        // For demonstration, we'll just update the framebuffer with a simple pattern
        
        static int frame = 0;
        frame++;
        
        // Generate a test pattern (color bars that animate)
        for (int y = 0; y < height; y++) {
            for (int x = 0; x < width; x++) {
                int section = (x * 8) / width;
                int color = 0;
                
                switch (section) {
                    case 0: color = 0xFF0000; break; // Red
                    case 1: color = 0x00FF00; break; // Green
                    case 2: color = 0x0000FF; break; // Blue
                    case 3: color = 0xFFFF00; break; // Yellow
                    case 4: color = 0xFF00FF; break; // Magenta
                    case 5: color = 0x00FFFF; break; // Cyan
                    case 6: color = 0xFFFFFF; break; // White
                    case 7: color = 0x000000; break; // Black
                }
                
                // Add some animation based on frame count
                int brightness = (255 - ((y + frame) % 256)) / 4;
                int r = ((color >> 16) & 0xFF) * brightness / 63;
                int g = ((color >> 8) & 0xFF) * brightness / 63;
                int b = (color & 0xFF) * brightness / 63;
                
                frameBuffer[y * width + x] = (r << 16) | (g << 8) | b;
            }
        }
        
        // In a real emulator, we'd also render text overlays, sprites, etc.
        // Add a text overlay "UltraHLE Style"
        renderText("UltraHLE Style", 20, 20, 0xFFFFFF);
        
        // Signal that a new frame is ready
        frameMutex.lock();
        frameReady = true;
        frameMutex.unlock();
        frameCV.notify_one();
    }
    
    // Get the current framebuffer as a QImage for display
    QImage getFramebufferImage() {
        std::unique_lock<std::mutex> lock(frameMutex);
        frameCV.wait(lock, [this]{ return frameReady; });
        
        // Convert the framebuffer to a QImage
        QImage image(reinterpret_cast<uchar*>(frameBuffer), width, height, 
                    width * sizeof(uint32_t), QImage::Format_RGB32);
        
        // Reset the frame ready flag
        frameReady = false;
        
        return image.copy(); // Return a copy to avoid threading issues
    }
    
    // Reset the framebuffer to black
    void clearFramebuffer() {
        std::memset(frameBuffer, 0, width * height * sizeof(uint32_t));
    }
    
    // Get dimensions
    int getWidth() const { return width; }
    int getHeight() const { return height; }
    
private:
    uint32_t* frameBuffer;
    int width;
    int height;
    
    std::mutex frameMutex;
    std::condition_variable frameCV;
    bool frameReady = false;
    
    // Simple text rendering function (very basic for demonstration)
    void renderText(const std::string& text, int x, int y, uint32_t color) {
        // This is a very simplified text renderer
        // In a real emulator, you'd use a proper font system
        
        const int charWidth = 8;
        const int charHeight = 8;
        
        // Simple bitmap font (just for demonstration)
        static const uint8_t font[128][8] = {
            // Space
            {0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00},
            // More characters would be defined here...
            // For simplicity, we'll just use a block for all characters
