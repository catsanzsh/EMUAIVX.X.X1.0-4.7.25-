// ========== Enhanced ROM Loading with Format Support ========== //
enum ROMFormat {
    UNKNOWN,
    Z64_BIG_ENDIAN,
    V64_BYTESWAPPED,
    N64_LITTLE_ENDIAN
};

ROMFormat detectROMFormat(const uint8_t* romData) {
    // Check first 4 bytes for known patterns
    if (romData[0] == 0x80 && romData[1] == 0x37) return Z64_BIG_ENDIAN;
    if (romData[0] == 0x37 && romData[1] == 0x80) return V64_BYTESWAPPED;
    if (romData[0] == 0x40 && romData[1] == 0x12) return N64_LITTLE_ENDIAN;
    return UNKNOWN;
}

void convertROMToNative(uint8_t* romData, size_t size, ROMFormat format) {
    switch(format) {
        case Z64_BIG_ENDIAN:
            // Already in correct format
            break;
        case V64_BYTESWAPPED:
            // Swap every 2 bytes
            for (size_t i = 0; i < size; i += 2) {
                std::swap(romData[i], romData[i+1]);
            }
            break;
        case N64_LITTLE_ENDIAN:
            // Swap every 4 bytes
            for (size_t i = 0; i < size; i += 4) {
                std::swap(romData[i], romData[i+3]);
                std::swap(romData[i+1], romData[i+2]);
            }
            break;
        default:
            throw std::runtime_error("Unsupported ROM format");
    }
}

// ========== Enhanced Dynarec with LRU Caching ========== //
class Dynarec {
    struct CacheEntry {
        RecompiledBlockFunc func;
        uint32_t lastUsed;
    };
    
    std::unordered_map<uint32_t, CacheEntry> blockCache;
    uint32_t currentTick = 0;
    static const size_t MAX_CACHE_SIZE = CACHE_SIZE;
    
public:
    RecompiledBlockFunc recompileBlock(uint32_t pc) {
        // LRU implementation
        auto it = blockCache.find(pc);
        if (it != blockCache.end()) {
            it->second.lastUsed = ++currentTick;
            cacheHits++;
            return it->second.func;
        }

        // Evict least recently used if cache full
        if (blockCache.size() >= MAX_CACHE_SIZE) {
            auto lru = std::min_element(blockCache.begin(), blockCache.end(),
                [](auto& a, auto& b) { return a.second.lastUsed < b.second.lastUsed; });
            blockCache.erase(lru);
        }

        // Recompile block (existing logic)
        auto func = /* existing recompilation logic */;
        blockCache[pc] = {func, ++currentTick};
        cacheMisses++;
        return func;
    }
};

// ========== UltraHLE-style Video Plugin ========== //
class UltraHLEVideo : public HLEVideoPlugin {
    struct RDPState {
        uint32_t textureAddress;
        uint32_t paletteAddress;
        // Other rendering state
    } rdpState;

public:
    void initialize() override {
        HLEVideoPlugin::initialize();
        std::cout << "[UltraHLE Video] Initialized with RDP HLE\n";
    }

    void processRDPCommands(uint32_t start, uint32_t end) {
        uint32_t ptr = start;
        while (ptr < end) {
            uint64_t cmd = read64(ptr);
            ptr += 8;
            
            const uint8_t cmdType = (cmd >> 56) & 0xFF;
            switch(cmdType) {
                case 0xF3: // Set Texture
                    rdpState.textureAddress = (cmd >> 41) & 0x00FFFFFF;
                    break;
                case 0xE4: // Draw Triangle
                    processTriangleCommand(cmd);
                    break;
                // Handle other commands
            }
        }
    }

private:
    uint64_t read64(uint32_t addr) {
        return *reinterpret_cast<uint64_t*>(gMem + addr);
    }

    void processTriangleCommand(uint64_t cmd) {
        // Extract vertex data and render using HLE
        Vertex v[3];
        // Parse vertex data from command
        for (int i = 0; i < 3; i++) {
            v[i].x = /* extract from cmd */;
            v[i].y = /* extract from cmd */;
            v[i].z = /* extract from cmd */;
            v[i].color = /* extract from cmd */;
        }
        renderTriangle(v[0], v[1], v[2]);
    }

    void renderTriangle(const Vertex& a, const Vertex& b, const Vertex& c) {
        // Simple software rasterization
        // In real implementation, use OpenGL/Direct3D
        QPainter painter(&framebuffer);
        painter.setPen(Qt::NoPen);
        QPoint points[3] = {{a.x, a.y}, {b.x, b.y}, {c.x, c.y}};
        painter.setBrush(QColor(a.color));
        painter.drawPolygon(points, 3);
    }
};

// ========== Enhanced Emulator Core ========== //
class EmulatorThread : public QThread {
    UltraHLEVideo videoPlugin;
    QTimer* renderTimer;

protected:
    void run() override {
        cpu.reset();
        cpu.running = true;
        cpu.pc = gROMHeader->pc;

        // Precompile initial blocks
        gDynarec.recompileBlock(cpu.pc);
        gDynarec.recompileBlock(cpu.pc + 4);

        while (!gStopRequested) {
            if (!gPaused) {
                gDynarec.runBlock(cpu.pc);
                videoPlugin.processRDPCommands(0x80000000, 0x80010000); // Example RDP range
                cpu.cycles += 2; // Account for RDP processing
            }
            QThread::usleep(1); // Yield
        }
    }

public:
    EmulatorThread(QObject* parent) : QThread(parent) {
        renderTimer = new QTimer(this);
        connect(renderTimer, &QTimer::timeout, [this]() {
            if (videoPlugin.frameReady())
                emit updateFrame(videoPlugin.getFramebufferImage());
        });
        renderTimer->start(16); // ~60 FPS
    }

signals:
    void updateFrame(const QImage& frame);
};

// ========== GUI Enhancements ========== //
class MainWindow : public QMainWindow {
    // Add format detection to file open dialog
    void openROM() {
        QString file = QFileDialog::getOpenFileName(this, "Open ROM", 
            "", "N64 ROMs (*.n64 *.v64 *.z64)");
        if (!file.isEmpty()) loadROM(file);
    }

    void loadROM(const QString& path) {
        QFile romFile(path);
        if (!romFile.open(QIODevice::ReadOnly)) return;

        QByteArray romData = romFile.readAll();
        ROMFormat format = detectROMFormat(
            reinterpret_cast<const uint8_t*>(romData.constData()));
        
        try {
            convertROMToNative(reinterpret_cast<uint8_t*>(romData.data()), 
                              romData.size(), format);
            std::memcpy(gMem, romData.constData(), std::min(romData.size(), MEMORY_SIZE));
            gROMHeader = reinterpret_cast<ROMHeader*>(gMem + 0x20);
            statusBar()->showMessage("Loaded: " + QString(gROMHeader->name));
        } catch (...) {
            statusBar()->showMessage("Invalid ROM format");
        }
    }
};
