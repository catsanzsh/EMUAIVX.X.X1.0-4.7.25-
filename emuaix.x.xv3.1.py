#!/usr/bin/env python3
"""
UltraHLE-Style N64 Emulator in Python + Tkinter
-----------------------------------------------
A high-level emulation approach inspired by UltraHLE's design philosophy
with modern Python techniques and improved architecture.
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox, Canvas
import threading
import time
import os
import struct
from enum import Enum, auto
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional, Union, ByteString
import logging
from PIL import Image, ImageTk
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("UltraPython")

# ========== N64 Constants ========== #
N64_CPU_FREQUENCY = 93_750_000  # Hz
N64_RDRAM_SIZE = 0x800000       # 8 MB
N64_PIF_ROM_SIZE = 0x1000       # 4 KB
N64_CART_ROM_ADDR = 0x10000000  # Start of cartridge ROM in N64 memory map
N64_RDRAM_ADDR = 0x00000000     # Start of RDRAM in N64 memory map

# ========== Enhanced ROM Handling ========== #
class ROMFormat(Enum):
    UNKNOWN = auto()
    Z64_BIG_ENDIAN = auto()
    V64_BYTESWAPPED = auto()
    N64_LITTLE_ENDIAN = auto()

@dataclass
class ROMInfo:
    """Stores information about a loaded ROM."""
    filename: str
    size: int
    format: ROMFormat
    internal_name: str
    country_code: str
    crc1: int
    crc2: int
    
    @staticmethod
    def from_data(filename: str, data: bytes) -> 'ROMInfo':
        """Extract ROM information from the header."""
        fmt = detect_rom_format(data)
        
        # Convert to big-endian format temporarily for header parsing
        tmp_data = bytearray(data[:0x1000])  # Just need the header
        if fmt != ROMFormat.Z64_BIG_ENDIAN:
            convert_rom_to_native(tmp_data, fmt)
        
        # Extract info from header
        internal_name = tmp_data[0x20:0x34].decode('ascii', errors='ignore').strip()
        country_code = chr(tmp_data[0x3E])
        crc1 = int.from_bytes(tmp_data[0x10:0x14], byteorder='big')
        crc2 = int.from_bytes(tmp_data[0x14:0x18], byteorder='big')
        
        return ROMInfo(
            filename=os.path.basename(filename),
            size=len(data),
            format=fmt,
            internal_name=internal_name,
            country_code=country_code,
            crc1=crc1,
            crc2=crc2
        )

def detect_rom_format(rom_data: bytes) -> ROMFormat:
    """Detect the format of an N64 ROM from its first 4 bytes."""
    if len(rom_data) < 4:
        return ROMFormat.UNKNOWN
    
    b0, b1, b2, b3 = rom_data[0], rom_data[1], rom_data[2], rom_data[3]

    # Check signatures:
    #  .z64: 0x80 0x37 0x12 0x40 (big-endian)
    #  .v64: 0x37 0x80 0x40 0x12 (byte-swapped)
    #  .n64: 0x40 0x12 0x37 0x80 (little-endian)
    if b0 == 0x80 and b1 == 0x37:
        return ROMFormat.Z64_BIG_ENDIAN
    if b0 == 0x37 and b1 == 0x80:
        return ROMFormat.V64_BYTESWAPPED
    if b0 == 0x40 and b1 == 0x12:
        return ROMFormat.N64_LITTLE_ENDIAN
    
    # Some ROMs might not have the exact signature, try an alternative method
    # by checking for ASCII text in the header area
    header_text = rom_data[0x20:0x34]
    if all(32 <= b <= 126 for b in header_text):
        # Looks like ASCII - assume it's Z64
        return ROMFormat.Z64_BIG_ENDIAN
    
    return ROMFormat.UNKNOWN

def convert_rom_to_native(rom_data: bytearray, fmt: ROMFormat):
    """
    Convert ROM data in-place to Z64 (big-endian) format if needed.
    This is the format used internally by the emulator.
    """
    size = len(rom_data)
    if fmt == ROMFormat.Z64_BIG_ENDIAN:
        # Already in correct format
        return
    elif fmt == ROMFormat.V64_BYTESWAPPED:
        # Swap every 2 bytes
        for i in range(0, size, 2):
            if i + 1 < size:
                rom_data[i], rom_data[i+1] = rom_data[i+1], rom_data[i]
    elif fmt == ROMFormat.N64_LITTLE_ENDIAN:
        # Swap every 4 bytes
        for i in range(0, size, 4):
            if i + 3 < size:
                rom_data[i], rom_data[i+3] = rom_data[i+3], rom_data[i]
                rom_data[i+1], rom_data[i+2] = rom_data[i+2], rom_data[i+1]
    else:
        raise ValueError("Unsupported or unknown ROM format")

# ========== Memory System ========== #
class MemoryDomain(Enum):
    RDRAM = auto()
    EXPANSION_PAK = auto()
    CARTRIDGE_ROM = auto()
    PIF_ROM = auto()
    PIF_RAM = auto()
    RSP_DMEM = auto()
    RSP_IMEM = auto()
    VI_REGISTERS = auto()
    AI_REGISTERS = auto()
    PI_REGISTERS = auto()
    RI_REGISTERS = auto()
    SI_REGISTERS = auto()
    MIPS_REGISTERS = auto()

class MemorySystem:
    """
    Manages N64 memory subsystems and provides read/write functions 
    with appropriate memory mapping.
    """
    def __init__(self):
        # Main memory components
        self.rdram = bytearray(N64_RDRAM_SIZE)  # 4 MB or 8 MB (with expansion)
        self.expansion_pak = None  # Additional 4MB if present
        self.cart_rom = None  # Will hold the ROM data
        self.pif_rom = bytearray(N64_PIF_ROM_SIZE)
        self.pif_ram = bytearray(64)  # PIF RAM for controller communication
        
        # RSP memories
        self.rsp_dmem = bytearray(4096)  # RSP data memory
        self.rsp_imem = bytearray(4096)  # RSP instruction memory
        
        # Hardware registers (simplified)
        self.vi_registers = bytearray(64)  # Video Interface
        self.ai_registers = bytearray(32)  # Audio Interface
        self.pi_registers = bytearray(48)  # Peripheral Interface
        self.ri_registers = bytearray(32)  # RDRAM Interface
        self.si_registers = bytearray(32)  # Serial Interface
        
        # CPU Registers
        self.cpu_registers = [0] * 32  # MIPS general-purpose registers
        self.cpu_cop0_registers = [0] * 32  # MIPS COP0 registers
        
        # Last memory access info for debugging
        self.last_access = {"address": 0, "value": 0, "type": "none"}
        
        logger.info("Memory system initialized")
    
    def map_address(self, address: int) -> Tuple[bytearray, int, MemoryDomain]:
        """
        Maps a physical address to the correct memory region.
        Returns (memory_array, offset, domain)
        """
        if 0x00000000 <= address < 0x03F00000:
            # RDRAM (including expansion)
            if address < N64_RDRAM_SIZE:
                return self.rdram, address, MemoryDomain.RDRAM
            elif self.expansion_pak and address < 0x00800000:
                return self.expansion_pak, address - 0x00400000, MemoryDomain.EXPANSION_PAK
        
        elif 0x10000000 <= address < 0x1FC00000:
            # Cartridge ROM
            if self.cart_rom and address - 0x10000000 < len(self.cart_rom):
                return self.cart_rom, address - 0x10000000, MemoryDomain.CARTRIDGE_ROM
        
        elif 0x1FC00000 <= address < 0x1FC01000:
            # PIF ROM
            return self.pif_rom, address - 0x1FC00000, MemoryDomain.PIF_ROM
        
        elif 0x1FC007C0 <= address < 0x1FC00800:
            # PIF RAM
            return self.pif_ram, address - 0x1FC007C0, MemoryDomain.PIF_RAM
        
        elif 0x04000000 <= address < 0x04001000:
            # RSP DMEM
            return self.rsp_dmem, address - 0x04000000, MemoryDomain.RSP_DMEM
        
        elif 0x04001000 <= address < 0x04002000:
            # RSP IMEM
            return self.rsp_imem, address - 0x04001000, MemoryDomain.RSP_IMEM
        
        elif 0x04400000 <= address < 0x04400040:
            # Video Interface registers
            return self.vi_registers, address - 0x04400000, MemoryDomain.VI_REGISTERS
        
        elif 0x04500000 <= address < 0x04500020:
            # Audio Interface registers
            return self.ai_registers, address - 0x04500000, MemoryDomain.AI_REGISTERS
        
        elif 0x04600000 <= address < 0x04600030:
            # Peripheral Interface registers
            return self.pi_registers, address - 0x04600000, MemoryDomain.PI_REGISTERS
        
        elif 0x04700000 <= address < 0x04700020:
            # RDRAM Interface registers
            return self.ri_registers, address - 0x04700000, MemoryDomain.RI_REGISTERS
        
        elif 0x04800000 <= address < 0x04800020:
            # Serial Interface registers
            return self.si_registers, address - 0x04800000, MemoryDomain.SI_REGISTERS
        
        # Unhandled memory address
        logger.warning(f"Unhandled memory access at 0x{address:08X}")
        return None, address, None
    
    def read8(self, address: int) -> int:
        """Read a byte from memory."""
        mem, offset, domain = self.map_address(address)
        if mem is None:
            return 0  # Unmapped memory returns 0
        
        value = mem[offset]
        self.last_access = {"address": address, "value": value, "type": "read8"}
        return value
    
    def read16(self, address: int) -> int:
        """Read a halfword (16 bits) from memory."""
        mem, offset, domain = self.map_address(address)
        if mem is None:
            return 0
        
        # Ensure proper alignment
        if offset + 1 >= len(mem):
            logger.warning(f"Out of bounds read16 at 0x{address:08X}")
            return 0
        
        value = (mem[offset] << 8) | mem[offset + 1]
        self.last_access = {"address": address, "value": value, "type": "read16"}
        return value
    
    def read32(self, address: int) -> int:
        """Read a word (32 bits) from memory."""
        mem, offset, domain = self.map_address(address)
        if mem is None:
            return 0
        
        # Ensure proper alignment
        if offset + 3 >= len(mem):
            logger.warning(f"Out of bounds read32 at 0x{address:08X}")
            return 0
        
        value = (mem[offset] << 24) | (mem[offset + 1] << 16) | (mem[offset + 2] << 8) | mem[offset + 3]
        self.last_access = {"address": address, "value": value, "type": "read32"}
        
        # Special handling for memory-mapped registers
        if domain in [MemoryDomain.VI_REGISTERS, MemoryDomain.AI_REGISTERS, 
                      MemoryDomain.PI_REGISTERS, MemoryDomain.RI_REGISTERS,
                      MemoryDomain.SI_REGISTERS]:
            # May need special handling for hardware registers
            pass
            
        return value
    
    def write8(self, address: int, value: int):
        """Write a byte to memory."""
        mem, offset, domain = self.map_address(address)
        if mem is None:
            return  # Write to unmapped memory is ignored
        
        # Read-only memory check
        if domain == MemoryDomain.CARTRIDGE_ROM or domain == MemoryDomain.PIF_ROM:
            logger.warning(f"Attempted write to read-only memory at 0x{address:08X}")
            return
        
        mem[offset] = value & 0xFF
        self.last_access = {"address": address, "value": value, "type": "write8"}
    
    def write16(self, address: int, value: int):
        """Write a halfword (16 bits) to memory."""
        mem, offset, domain = self.map_address(address)
        if mem is None:
            return
        
        # Read-only memory check
        if domain == MemoryDomain.CARTRIDGE_ROM or domain == MemoryDomain.PIF_ROM:
            logger.warning(f"Attempted write to read-only memory at 0x{address:08X}")
            return
        
        # Ensure proper alignment
        if offset + 1 >= len(mem):
            logger.warning(f"Out of bounds write16 at 0x{address:08X}")
            return
        
        mem[offset] = (value >> 8) & 0xFF
        mem[offset + 1] = value & 0xFF
        self.last_access = {"address": address, "value": value, "type": "write16"}
    
    def write32(self, address: int, value: int):
        """Write a word (32 bits) to memory."""
        mem, offset, domain = self.map_address(address)
        if mem is None:
            return
        
        # Read-only memory check
        if domain == MemoryDomain.CARTRIDGE_ROM or domain == MemoryDomain.PIF_ROM:
            logger.warning(f"Attempted write to read-only memory at 0x{address:08X}")
            return
        
        # Ensure proper alignment
        if offset + 3 >= len(mem):
            logger.warning(f"Out of bounds write32 at 0x{address:08X}")
            return
        
        mem[offset] = (value >> 24) & 0xFF
        mem[offset + 1] = (value >> 16) & 0xFF
        mem[offset + 2] = (value >> 8) & 0xFF
        mem[offset + 3] = value & 0xFF
        self.last_access = {"address": address, "value": value, "type": "write32"}
        
        # Special handling for memory-mapped registers
        if domain == MemoryDomain.VI_REGISTERS:
            # Video Interface Register write - might trigger display updates
            if (address & 0xFF) == 0x10:  # VI_CURRENT_REG
                # Handle vertical interrupt
                pass
        elif domain == MemoryDomain.AI_REGISTERS:
            # Audio Interface Register
            pass
        elif domain == MemoryDomain.PI_REGISTERS:
            # Peripheral Interface - might trigger DMA
            self._handle_pi_register_write(address, value)
    
    def _handle_pi_register_write(self, address: int, value: int):
        """Handle special cases for Peripheral Interface register writes."""
        reg_offset = address & 0xFF
        
        if reg_offset == 0x00:  # PI_DRAM_ADDR_REG
            self.pi_dram_addr = value & 0x00FFFFF8  # Mask and align
        elif reg_offset == 0x04:  # PI_CART_ADDR_REG
            self.pi_cart_addr = value & 0x1FFFFFF8  # Mask and align
        elif reg_offset == 0x08:  # PI_RD_LEN_REG
            # DMA from cart to RDRAM
            self._do_pi_dma(self.pi_dram_addr, self.pi_cart_addr, value & 0x00FFFFFF)
        elif reg_offset == 0x0C:  # PI_WR_LEN_REG
            # DMA from RDRAM to cart (e.g., save data)
            # Not fully implemented for simplicity
            pass
        elif reg_offset == 0x10:  # PI_STATUS_REG
            # Clear interrupt if bit 1 is set
            if value & 0x02:
                # Clear PI interrupt
                pass
    
    def _do_pi_dma(self, rdram_addr: int, cart_addr: int, length: int):
        """Perform DMA transfer from cartridge ROM to RDRAM."""
        # Adjust length (N64 DMA length is encoded as length-1)
        real_length = (length & 0x00FFFFFF) + 1
        
        # Make sure cart address is in ROM range
        if 0x10000000 <= cart_addr < 0x1FC00000:
            cart_offset = cart_addr - 0x10000000
            
            # Check bounds
            if cart_offset + real_length <= len(self.cart_rom):
                # Copy data
                for i in range(real_length):
                    if rdram_addr + i < len(self.rdram):
                        self.rdram[rdram_addr + i] = self.cart_rom[cart_offset + i]
                
                logger.debug(f"DMA: Copied {real_length} bytes from "
                           f"ROM+0x{cart_offset:X} to RDRAM+0x{rdram_addr:X}")
            else:
                logger.warning(f"DMA: Out of bounds cart access: "
                             f"ROM+0x{cart_offset:X} + {real_length} bytes")
        else:
            logger.warning(f"DMA: Invalid cart address: 0x{cart_addr:08X}")
    
    def load_rom(self, rom_data: bytes):
        """Load ROM data into cartridge memory."""
        self.cart_rom = bytearray(rom_data)
        logger.info(f"ROM loaded: {len(rom_data)} bytes")

# ========== R4300i CPU Emulation ========== #
class CPUOperation(Enum):
    """Supported CPU operations for the MIPS R4300i."""
    NOP = auto()
    ADD = auto()
    ADDI = auto()
    ADDIU = auto()
    ADDU = auto()
    AND = auto()
    ANDI = auto()
    BEQ = auto()
    BGEZ = auto()
    BGEZAL = auto()
    BGTZ = auto()
    BLEZ = auto()
    BLTZ = auto()
    BLTZAL = auto()
    BNE = auto()
    BREAK = auto()
    CACHE = auto()
    COP0 = auto()
    COP1 = auto()
    DADD = auto()
    DADDI = auto()
    DADDIU = auto()
    DADDU = auto()
    DIV = auto()
    DIVU = auto()
    DMULT = auto()
    DMULTU = auto()
    DSLL = auto()
    DSLLV = auto()
    DSRA = auto()
    DSRAV = auto()
    DSRL = auto()
    DSRLV = auto()
    DSUB = auto()
    DSUBU = auto()
    J = auto()
    JAL = auto()
    JALR = auto()
    JR = auto()
    LB = auto()
    LBU = auto()
    LD = auto()
    LDL = auto()
    LDR = auto()
    LH = auto()
    LHU = auto()
    LL = auto()
    LLD = auto()
    LUI = auto()
    LW = auto()
    LWL = auto()
    LWR = auto()
    LWU = auto()
    MFC0 = auto()
    MFC1 = auto()
    MFHI = auto()
    MFLO = auto()
    MTC0 = auto()
    MTC1 = auto()
    MTHI = auto()
    MTLO = auto()
    MULT = auto()
    MULTU = auto()
    OR = auto()
    ORI = auto()
    SB = auto()
    SC = auto()
    SCD = auto()
    SD = auto()
    SDL = auto()
    SDR = auto()
    SH = auto()
    SLL = auto()
    SLLV = auto()
    SLT = auto()
    SLTI = auto()
    SLTIU = auto()
    SLTU = auto()
    SRA = auto()
    SRAV = auto()
    SRL = auto()
    SRLV = auto()
    SUB = auto()
    SUBU = auto()
    SW = auto()
    SWL = auto()
    SWR = auto()
    SYNC = auto()
    SYSCALL = auto()
    TEQ = auto()
    TGE = auto()
    TGEI = auto()
    TGEIU = auto()
    TGEU = auto()
    TLT = auto()
    TLTI = auto()
    TLTIU = auto()
    TLTU = auto()
    TNE = auto()
    TNEI = auto()
    XOR = auto()
    XORI = auto()

class Instruction:
    """Represents a decoded R4300i instruction."""
    def __init__(self, opcode: int):
        self.opcode = opcode
        self.operation = CPUOperation.NOP
        self.rs = (opcode >> 21) & 0x1F
        self.rt = (opcode >> 16) & 0x1F
        self.rd = (opcode >> 11) & 0x1F
        self.shamt = (opcode >> 6) & 0x1F
        self.funct = opcode & 0x3F
        self.immediate = opcode & 0xFFFF
        self.address = opcode & 0x3FFFFFF
        
        # Sign-extend immediate if needed
        if self.immediate & 0x8000:
            self.immediate = -((self.immediate ^ 0xFFFF) + 1)
        
        # Decode instruction
        self._decode()
    
    def _decode(self):
        """Decode the instruction based on opcode fields."""
        op = (self.opcode >> 26) & 0x3F
        
        if op == 0:  # SPECIAL
            self._decode_special()
        elif op == 1:  # REGIMM
            self._decode_regimm()
        elif op == 2:  # J
            self.operation = CPUOperation.J
        elif op == 3:  # JAL
            self.operation = CPUOperation.JAL
        elif op == 4:  # BEQ
            self.operation = CPUOperation.BEQ
        elif op == 5:  # BNE
            self.operation = CPUOperation.BNE
        elif op == 6:  # BLEZ
            self.operation = CPUOperation.BLEZ
        elif op == 7:  # BGTZ
            self.operation = CPUOperation.BGTZ
        elif op == 8:  # ADDI
            self.operation = CPUOperation.ADDI
        elif op == 9:  # ADDIU
            self.operation = CPUOperation.ADDIU
        elif op == 10:  # SLTI
            self.operation = CPUOperation.SLTI
        elif op == 11:  # SLTIU
            self.operation = CPUOperation.SLTIU
        elif op == 12:  # ANDI
            self.operation = CPUOperation.ANDI
        elif op == 13:  # ORI
            self.operation = CPUOperation.ORI
        elif op == 14:  # XORI
            self.operation = CPUOperation.XORI
        elif op == 15:  # LUI
            self.operation = CPUOperation.LUI
        elif op == 16:  # COP0
            self.operation = CPUOperation.COP0
        elif op == 17:  # COP1
            self.operation = CPUOperation.COP1
        elif op == 32:  # LB
            self.operation = CPUOperation.LB
        elif op == 33:  # LH
            self.operation = CPUOperation.LH
        elif op == 35:  # LW
            self.operation = CPUOperation.LW
        elif op == 36:  # LBU
            self.operation = CPUOperation.LBU
        elif op == 37:  # LHU
            self.operation = CPUOperation.LHU
        elif op == 39:  # LWU
            self.operation = CPUOperation.LWU
        elif op == 40:  # SB
            self.operation = CPUOperation.SB
        elif op == 41:  # SH
            self.operation = CPUOperation.SH
        elif op == 43:  # SW
            self.operation = CPUOperation.SW
        elif op == 49:  # LWC1
            self.operation = CPUOperation.LWC1
        elif op == 57:  # SWC1
            self.operation = CPUOperation.SWC1
        else:
            logger.warning(f"Unimplemented opcode: 0x{op:02X}")
    
    def _decode_special(self):
        """Decode SPECIAL instructions (opcode 0)."""
        if self.funct == 0:  # SLL
            self.operation = CPUOperation.SLL
        elif self.funct == 2:  # SRL
            self.operation = CPUOperation.SRL
        elif self.funct == 3:  # SRA
            self.operation = CPUOperation.SRA
        elif self.funct == 4:  # SLLV
            self.operation = CPUOperation.SLLV
        elif self.funct == 6:  # SRLV
            self.operation = CPUOperation.SRLV
        elif self.funct == 7:  # SRAV
            self.operation = CPUOperation.SRAV
        elif self.funct == 8:  # JR
            self.operation = CPUOperation.JR
        elif self.funct == 9:  # JALR
            self.operation = CPUOperation.JALR
        elif self.funct == 12:  # SYSCALL
            self.operation = CPUOperation.SYSCALL
        elif self.funct == 13:  # BREAK
            self.operation = CPUOperation.BREAK
        elif self.funct == 16:  # MFHI
            self.operation = CPUOperation.MFHI
        elif self.funct == 17:  # MTHI
            self.operation = CPUOperation.MTHI
        elif self.funct == 18:  # MFLO
            self.operation = CPUOperation.MFLO
        elif self.funct == 19:  # MTLO
            self.operation = CPUOperation.MTLO
        elif self.funct == 24:  # MULT
            self.operation = CPUOperation.MULT
        elif self.funct == 25:  # MULTU
            self.operation = CPUOperation.MULTU
        elif self.funct == 26:  # DIV
            self.operation = CPUOperation.DIV
        elif self.funct == 27:  # DIVU
            self.operation = CPUOperation.DIVU
        elif self.funct == 32:  # ADD
            self.operation = CPUOperation.ADD
        elif self.funct == 33:  # ADDU
            self.operation = CPUOperation.ADDU
        elif self.funct == 34:  # SUB
            self.operation = CPUOperation.SUB
        elif self.funct == 35:  # SUBU
            self.operation = CPUOperation.SUBU
        elif self.funct == 36:  # AND
            self.operation = CPUOperation.AND
        elif self.funct == 37:  # OR
            self.operation = CPUOperation.OR
        elif self.funct == 38:  # XOR
            self.operation = CPUOperation.XOR
        elif self.funct == 42:  # SLT
            self.operation = CPUOperation.SLT
        elif self.funct == 43:  # SLTU
            self.operation = CPUOperation.SLTU
        else:
            logger.warning(f"Unimplemented SPECIAL instruction: 0x{self.funct:02X}")
    
    def _decode_regimm(self):
        """Decode REGIMM instructions (opcode 1)."""
        rt = self.rt
        
        if rt == 0:  # BLTZ
            self.operation = CPUOperation.BLTZ
        elif rt == 1:  # BGEZ
            self.operation = CPUOperation.BGEZ
        elif rt == 16:  # BLTZAL
            self.operation = CPUOperation.BLTZAL
        elif rt == 17:  # BGEZAL
            self.operation = CPUOperation.BGEZAL
        else:
            logger.warning(f"Unimplemented REGIMM instruction: 0x{rt:02X}")
    
    def __str__(self):
        """Return a human-readable representation of the instruction."""
        return f"{self.operation.name}: rs={self.rs}, rt={self.rt}, rd={self.rd}, " \
               f"imm=0x{self.immediate & 0xFFFF:04X}, addr=0x{self.address:07X}"

class CPU:
    """Emulates the R4300i CPU."""
    def __init__(self, memory: MemorySystem):
        self.memory = memory
        
        # CPU registers
        self.registers = [0] * 32  # General purpose registers (R0-R31)
        self.pc = 0  # Program counter
        self.hi = 0  # Multiply/divide high result
        self.lo = 0  # Multiply/divide low result
        
        # FPU registers (simplified)
        self.fpu_registers = [0.0] * 32
        
        # COP0 registers (system control)
        self.cop0_registers = [0] * 32
        
        # Control state
        self.delay_slot = False
        self.next_pc = 0
        self.delay_slot_pc = 0
        
        # Statistics
        self.instructions_executed = 0
        
        logger.info("CPU initialized")
    
    def reset(self):
        """Reset the CPU to its initial state."""
        self.registers = [0] * 32
        self.pc = 0xBFC00000  # Start at PIF ROM
        self.hi = 0
        self.lo = 0
        self.fpu_registers = [0.0] * 32
        self.cop0_registers = [0] * 32
        self.delay_slot = False
        self.next_pc = 0
        self.delay_slot_pc = 0
        self.instructions_executed = 0
        
        # Initialize stack pointer (often done in boot code, but we'll set it directly)
        self.registers[29] = 0x80000400  # SP
        
        logger.info("CPU reset")
    
    def fetch(self) -> int:
        """Fetch an instruction from memory at the current PC."""
        try:
            return self.memory.read32(self.pc)
        except Exception as e:
            logger.error(f"Error fetching instruction at PC=0x{self.pc:08X}: {e}")
            return 0
    
    def decode(self, opcode: int) -> Instruction:
        """Decode an instruction."""
        return Instruction(opcode)
    
    def execute(self, instruction: Instruction):
        """Execute an instruction."""
        # Skip execution for register 0 (always 0)
        r0_write = (instruction.operation in [
            CPUOperation.ADD, CPUOperation.ADDU, CPUOperation.SUB, CPUOperation.SUBU,
            CPUOperation.AND, CPUOperation.OR, CPUOperation.XOR, CPUOperation.NOR,
            CPUOperation.SLT, CPUOperation.SLTU
        ] and instruction.rd == 0)
        
        if r0_write:
            logger.debug(f"Ignoring write to R0 at PC=0x{self.pc:08X}")
            return
        
        # Advance PC (unless modified by instruction)
        if not self.delay_slot:
            self.next_pc = self.pc + 4
        else:
            # We're in a delay slot, use the saved next_pc
            self.delay_slot = False
        
        # Execute based on operation
        op = instruction.operation
        rs = instruction.rs
        rt = instruction.rt
        rd = instruction.rd
        imm = instruction.immediate
        addr = instruction.address
        shamt = instruction.shamt
        
        rs_val = self.registers[rs] if rs < 32 else 0
        rt_val = self.registers[rt] if rt < 32 else 0
        
        # Execute operation
        if op == CPUOperation.NOP:
            pass  # No operation
        
        # Arithmetic operations
        elif op == CPUOperation.ADD or op == CPUOperation.ADDU:
            self.registers[rd] = (rs_val + rt_val) & 0xFFFFFFFF
        
        elif op == CPUOperation.ADDI or op == CPUOperation.ADDIU:
            self.registers[rt] = (rs_val + imm) & 0xFFFFFFFF
        
        elif op == CPUOperation.SUB or op == CPUOperation.SUBU:
            self.registers[rd] = (rs_val - rt_val) & 0xFFFFFFFF
        
        # Logical operations
        elif op == CPUOperation.AND:
            self.registers[rd] = rs_val & rt_val
        
        elif op == CPUOperation.ANDI:
            self.registers[rt] = rs_val & (imm & 0xFFFF)  # Zero-extended immediate
        
        elif op == CPUOperation.OR:
            self.registers[rd] = rs_val | rt_val
        
        elif op == CPUOperation.ORI:
            self.registers[rt] = rs_val | (imm & 0xFFFF)  # Zero-extended immediate
        
        elif op == CPUOperation.XOR:
            self.registers[rd] = rs_val ^ rt_val
        
        elif op == CPUOperation.XORI:
            self.registers[rt] = rs_val ^ (imm & 0xFFFF)  # Zero-extended immediate
        
        # Shifts
        elif op == CPUOperation.SLL:
            self.registers[rd] = (rt_val << shamt) & 0xFFFFFFFF
        
        elif op == CPUOperation.SRL:
            self.registers[rd] = (rt_val >> shamt) & 0xFFFFFFFF
        
        elif op == CPUOperation.SRA:
            # Arithmetic shift (sign extending)
            self.registers[rd] = ((rt_val & 0xFFFFFFFF) >> shamt) & 0xFFFFFFFF
            if rt_val & 0x80000000:  # Sign bit is set
                mask = ((1 << shamt) - 1) << (32 - shamt)
                self.registers[rd] |= mask
        
        # Memory operations
        elif op == CPUOperation.LW:
            addr = (rs_val + imm) & 0xFFFFFFFF
            self.registers[rt] = self.memory.read32(addr)
        
        elif op == CPUOperation.SW:
            addr = (rs_val + imm) & 0xFFFFFFFF
            self.memory.write32(addr, rt_val)
        
        elif op == CPUOperation.LB:
            addr = (rs_val + imm) & 0xFFFFFFFF
            value = self.memory.read8(addr)
            # Sign extend byte to 32 bits
            if value & 0x80:
                value |= 0xFFFFFF00
            self.registers[rt] = value
        
        elif op == CPUOperation.LBU:
            addr = (rs_val + imm) & 0xFFFFFFFF
            value = self.memory.read8(addr) & 0xFF  # Zero extend
            self.registers[rt] = value
        
        elif op == CPUOperation.SB:
            addr = (rs_val + imm) & 0xFFFFFFFF
            self.memory.write8(addr, rt_val & 0xFF)
        
        elif op == CPUOperation.LH:
            addr = (rs_val + imm) & 0xFFFFFFFF
            value = self.memory.read16(addr)
            # Sign extend halfword to 32 bits
            if value & 0x8000:
                value |= 0xFFFF0000
            self.registers[rt] = value
        
        elif op == CPUOperation.LHU:
            addr = (rs_val + imm) & 0xFFFFFFFF
            value = self.memory.read16(addr) & 0xFFFF  # Zero extend
            self.registers[rt] = value
        
        elif op == CPUOperation.SH:
            addr = (rs_val + imm) & 0xFFFFFFFF
            self.memory.write16(addr, rt_val & 0xFFFF)
        
        # Jump and branch operations
        elif op == CPUOperation.J:
            target = (self.pc & 0xF0000000) | (addr << 2)
            # We need to execute one more instruction before jump (delay slot)
            self.delay_slot = True
            self.delay_slot_pc = self.pc + 4
            self.next_pc = target
        
        elif op == CPUOperation.JAL:
            target = (self.pc & 0xF0000000) | (addr << 2)
            self.registers[31] = self.pc + 8  # Link register = PC + 8
            # We need to execute one more instruction before jump (delay slot)
            self.delay_slot = True
            self.delay_slot_pc = self.pc + 4
            self.next_pc = target
        
        elif op == CPUOperation.JR:
            target = rs_val
            # We need to execute one more instruction before jump (delay slot)
            self.delay_slot = True
            self.delay_slot_pc = self.pc + 4
            self.next_pc = target
        
        elif op == CPUOperation.BEQ:
            if rs_val == rt_val:
                target = self.pc + 4 + (imm << 2)
                # We need to execute one more instruction before branch (delay slot)
                self.delay_slot = True
                self.delay_slot_pc = self.pc + 4
                self.next_pc = target
        
        elif op == CPUOperation.BNE:
            if rs_val != rt_val:
                target = self.pc + 4 + (imm << 2)
                # We need to execute one more instruction before branch (delay slot)
                self.delay_slot = True
                self.delay_slot_pc = self.pc + 4
                self.next_pc = target
        
        # Comparison operations
        elif op == CPUOperation.SLT:
            # Signed comparison
            self.registers[rd] = 1 if ((rs_val & 0x80000000) != (rt_val & 0x80000000)) ? \
                                     (rs_val & 0x80000000) : (rs_val < rt_val) else 0
        
        elif op == CPUOperation.SLTU:
            # Unsigned comparison
            self.registers[rd] = 1 if (rs_val & 0xFFFFFFFF) < (rt_val & 0xFFFFFFFF) else 0
        
        # Load immediate
        elif op == CPUOperation.LUI:
            self.registers[rt] = (imm << 16) & 0xFFFFFFFF
        
        else:
            logger.warning(f"Unimplemented operation: {op} at PC=0x{self.pc:08X}")
        
        # Ensure R0 is always 0
        self.registers[0] = 0
        
        # Update PC
        if self.delay_slot:
            self.pc = self.delay_slot_pc
        else:
            self.pc = self.next_pc
        
        # Update stats
        self.instructions_executed += 1
    
    def step(self):
        """Execute a single instruction."""
        opcode = self.fetch()
        instruction = self.decode(opcode)
        self.execute(instruction)
        return instruction

# ========== UltraHLE-style Video Plugin ========== #
class VideoCommand(Enum):
    """RDP commands supported by the video plugin."""
    FILL_TRIANGLE = auto()
    FILL_RECTANGLE = auto()
    TEXTURE_RECTANGLE = auto()
    LOAD_BLOCK = auto()
    LOAD_TILE = auto()
    SET_TILE = auto()
    SET_TILE_SIZE = auto()
    SET_COLOR_IMAGE = auto()
    SET_FILL_COLOR = auto()
    SET_FOG_COLOR = auto()
    SET_BLEND_COLOR = auto()
    SET_PRIM_COLOR = auto()
    SET_ENV_COLOR = auto()
    SET_COMBINE = auto()
    SYNC_FULL = auto()
    SYNC_PIPE = auto()
    SYNC_LOAD = auto()
    UNKNOWN = auto()

class UltraHLEVideo:
    """
    UltraHLE-style video plugin that performs high-level emulation
    of the RDP commands rather than low-level pixel-by-pixel processing.
    """
    def __init__(self, memory: MemorySystem):
        self.memory = memory
        self.width = 640
        self.height = 480
        self.framebuffer = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        self.frame_ready = False
        self.frame_data = None
        self.frame_count = 0
        
        # RDP state
        self.color_image_address = 0
        self.color_image_width = 0
        self.color_image_format = 0
        self.color_image_size = 0
        self.fill_color = 0
        self.blend_color = 0
        self.prim_color = 0
        self.env_color = 0
        self.fog_color = 0
        self.combine_mode = 0
        
        # Tile state
        self.tiles = [{} for _ in range(8)]  # 8 tile descriptors
        
        logger.info("UltraHLE Video plugin initialized")
    
    def initialize(self):
        """Initialize the video plugin."""
        logger.info("UltraHLE Video plugin initialized with RDP HLE")
    
    def process_rdp_commands(self, start_address: int, end_address: int):
        """
        Process RDP commands from memory.
        
        In UltraHLE style, this identifies and processes higher-level graphics
        functions rather than emulating the RDP at the microcode level.
        """
        address = start_address
        commands_processed = 0
        
        # In a real implementation, we would parse actual RDP commands from memory
        # This is a simplified version that just illustrates the concept
        while address < end_address and commands_processed < 100:
            # Read command and length
            cmd_word = self.memory.read32(address)
            cmd_id = (cmd_word >> 24) & 0xFF
            
            if cmd_id == 0xF1:  # Example: Set Color Image
                self._set_color_image(address)
                address += 8
            elif cmd_id == 0xF7:  # Example: Fill Rectangle
                self._fill_rectangle(address)
                address += 8
            else:
                # Unrecognized command, just skip ahead
                address += 8
            
            commands_processed += 1
        
        # In a real implementation, we'd do actual rendering here
        # For now, we'll just set the frame ready flag
        self.frame_ready = True
        self.frame_count += 1
    
    def _set_color_image(self, address: int):
        """Set the color image (framebuffer) parameters."""
        cmd_word0 = self.memory.read32(address)
        cmd_word1 = self.memory.read32(address + 4)
        
        self.color_image_format = (cmd_word0 >> 21) & 0x7
        self.color_image_size = (cmd_word0 >> 19) & 0x3
        self.color_image_width = (cmd_word0 & 0x3FF) + 1
        self.color_image_address = cmd_word1 & 0xFFFFFFFF
        
        logger.debug(f"Set Color Image: addr=0x{self.color_image_address:08X}, "
                   f"width={self.color_image_width}, format={self.color_image_format}")
    
    def _fill_rectangle(self, address: int):
        """Process a fill rectangle command."""
        cmd_word0 = self.memory.read32(address)
        cmd_word1 = self.memory.read32(address + 4)
        
        # Extract rectangle coordinates (as 10.2 fixed point)
        x0 = ((cmd_word1 >> 12) & 0xFFF) >> 2
        y0 = (cmd_word1 & 0xFFF) >> 2
        x1 = ((cmd_word0 >> 12) & 0xFFF) >> 2
        y1 = (cmd_word0 & 0xFFF) >> 2
        
        # Apply fill to framebuffer
        color = (self.fill_color >> 8) & 0xFFFFFF  # 24-bit color
        r = (color >> 16) & 0xFF
        g = (color >> 8) & 0xFF
        b = color & 0xFF
        
        # Convert coordinates to fit within our framebuffer
        x0 = max(0, min(x0, self.width - 1))
        y0 = max(0, min(y0, self.height - 1))
        x1 = max(0, min(x1, self.width - 1))
        y1 = max(0, min(y1, self.height - 1))
        
        # Fill the rectangle in our framebuffer
        if x1 > x0 and y1 > y0:
            self.framebuffer[y0:y1, x0:x1, 0] = r
            self.framebuffer[y0:y1, x0:x1, 1] = g
            self.framebuffer[y0:y1, x0:x1, 2] = b
        
        logger.debug(f"Fill Rectangle: ({x0},{y0})-({x1},{y1}) color=0x{color:06X}")
    
    def get_framebuffer_image(self):
        """Return the current framebuffer as a PIL Image."""
        return Image.fromarray(self.framebuffer)
    
    def is_frame_ready(self) -> bool:
        """Check if a new frame is ready to be displayed."""
        return self.frame_ready
    
    def reset_frame_flag(self):
        """Reset the frame ready flag after displaying it."""
        self.frame_ready = False

# ========== UltraHLE-style Audio Plugin ========== #
class UltraHLEAudio:
    """
    UltraHLE-style audio plugin that performs high-level emulation
    of the audio subsystem.
    """
    def __init__(self, memory: MemorySystem):
        self.memory = memory
        self.sample_rate = 44100
        self.channels = 2
        self.buffer_size = 4096
        self.audio_buffer = np.zeros((self.buffer_size, self.channels), dtype=np.int16)
        self.audio_position = 0
        self.dma_count = 0
        self.ai_dram_addr = 0
        self.ai_length = 0
        self.ai_control = 0
        
        logger.info("UltraHLE Audio plugin initialized")
    
    def initialize(self):
        """Initialize the audio plugin."""
        # In a real implementation, this would set up audio output
        pass
    
    def process_audio_commands(self):
        """
        Process AI (Audio Interface) commands.
        
        UltraHLE style: Instead of emulating the audio hardware at cycle level,
        we look for DMA transfers and high-level commands.
        """
        # Check AI registers
        control = self.memory.ai_registers[0x08]  # AI_CONTROL_REG
        dram_addr = (self.memory.ai_registers[0x00] << 24) | \
                    (self.memory.ai_registers[0x01] << 16) | \
                    (self.memory.ai_registers[0x02] << 8) | \
                    (self.memory.ai_registers[0x03])  # AI_DRAM_ADDR_REG
        length = (self.memory.ai_registers[0x04] << 24) | \
                 (self.memory.ai_registers[0x05] << 16) | \
                 (self.memory.ai_registers[0x06] << 8) | \
                 (self.memory.ai_registers[0x07])  # AI_LEN_REG
        
        # If DMA is active and we have a new audio buffer
        if control & 0x01 and dram_addr != self.ai_dram_addr and length > 0:
            self._process_audio_dma(dram_addr, length)
            self.ai_dram_addr = dram_addr
            self.ai_length = length
    
    def _process_audio_dma(self, dram_addr: int, length: int):
        """Process audio DMA transfer from RDRAM to audio buffer."""
        # In a real implementation, this would read audio data from RDRAM
        # and convert it to PCM samples for output
        
        # Simplified example: just increment DMA count
        self.dma_count += 1
        logger.debug(f"Audio DMA #{self.dma_count}: addr=0x{dram_addr:08X}, len={length}")
    
    def get_audio_stats(self) -> dict:
        """Return statistics about audio processing."""
        return {
            "dma_count": self.dma_count,
            "last_addr": self.ai_dram_addr,
            "last_length": self.ai_length
        }

# ========== Input Plugin ========== #
class ControllerState:
    """Stores the state of an N64 controller."""
    def __init__(self):
        self.buttons = 0  # Button state (bit flags)
        self.x_axis = 0   # Analog stick X position (-80 to +80)
        self.y_axis = 0   # Analog stick Y position (-80 to +80)
        
        # Button constants
        self.A_BUTTON = 0x8000
        self.B_BUTTON = 0x4000
        self.Z_BUTTON = 0x2000
        self.START_BUTTON = 0x1000
        self.DPAD_UP = 0x0800
        self.DPAD_DOWN = 0x0400
        self.DPAD_LEFT = 0x0200
        self.DPAD_RIGHT = 0x0100
        self.L_BUTTON = 0x0020
        self.R_BUTTON = 0x0010
        self.C_UP = 0x0008
        self.C_DOWN = 0x0004
        self.C_LEFT = 0x0002
        self.C_RIGHT = 0x0001
    
    def set_button(self, button: int, pressed: bool):
        """Set the state of a button."""
        if pressed:
            self.buttons |= button
        else:
            self.buttons &= ~button
    
    def set_analog_stick(self, x: int, y: int):
        """Set the position of the analog stick."""
        self.x_axis = max(-80, min(80, x))
        self.y_axis = max(-80, min(80, y))
    
    def get_state(self) -> bytes:
        """Return the controller state in N64 format."""
        # Format: A0 00 xx yy
        # A0 = first byte, upper 4 bits are 1010 (controller present)
        # 00 = buttons high byte then low byte
        # xx = analog stick X position
        # yy = analog stick Y position
        return bytes([
            0xA0,
            (self.buttons >> 8) & 0xFF,
            self.buttons & 0xFF,
            self.x_axis & 0xFF,
            self.y_axis & 0xFF
        ])

class InputPlugin:
    """Handles input from controllers and translates them to N64 format."""
    def __init__(self, memory: MemorySystem):
        self.memory = memory
        self.controllers = [ControllerState() for _ in range(4)]
        
        # Key mappings (sample for player 1)
        self.key_mapping = {
            # Keyboard key: (controller number, button or axis, value)
            'Up': (0, 'DPAD_UP', True),
            'Down': (0, 'DPAD_DOWN', True),
            'Left': (0, 'DPAD_LEFT', True),
            'Right': (0, 'DPAD_RIGHT', True),
            'z': (0, 'A_BUTTON', True),
            'x': (0, 'B_BUTTON', True),
            'c': (0, 'Z_BUTTON', True),
            'Return': (0, 'START_BUTTON', True),
            'a': (0, 'C_LEFT', True),
            's': (0, 'C_DOWN', True),
            'd': (0, 'C_RIGHT', True),
            'w': (0, 'C_UP', True),
            'q': (0, 'L_BUTTON', True),
            'e': (0, 'R_BUTTON', True),
        }
        
        logger.info("Input plugin initialized")
    
    def process_key_event(self, key: str, pressed: bool):
        """Process a keyboard event and update controller state."""
        if key in self.key_mapping:
            controller_num, button_or_axis, value = self.key_mapping[key]
            
            # Handle buttons
            if hasattr(self.controllers[controller_num], button_or_axis):
                button = getattr(self.controllers[controller_num], button_or_axis)
                self.controllers[controller_num].set_button(button, pressed)
            
            # Handle analog stick simulation with arrow keys
            if button_or_axis == 'DPAD_UP' and pressed:
                self.controllers[controller_num].set_analog_stick(
                    self.controllers[controller_num].x_axis, 80)
            elif button_or_axis == 'DPAD_DOWN' and pressed:
                self.controllers[controller_num].set_analog_stick(
                    self.controllers[controller_num].x_axis, -80)
            elif button_or_axis == 'DPAD_LEFT' and pressed:
                self.controllers[controller_num].set_analog_stick(
                    -80, self.controllers[controller_num].y_axis)
            elif button_or_axis == 'DPAD_RIGHT' and pressed:
                self.controllers[controller_num].set_analog_stick(
                    80, self.controllers[controller_num].y_axis)
            
            # Reset analog stick when direction keys are released
            if not pressed and button_or_axis in ['DPAD_UP', 'DPAD_DOWN']:
                self.controllers[controller_num].set_analog_stick(
                    self.controllers[controller_num].x_axis, 0)
            if not pressed and button_or_axis in ['DPAD_LEFT', 'DPAD_RIGHT']:
                self.controllers[controller_num].set_analog_stick(
                    0, self.controllers[controller_num].y_axis)
    
    def update_pif_ram(self):
        """
        Update PIF RAM with controller data.
        This is called when the CPU reads or writes to PIF RAM.
        """
        # Check if the PIF RAM contains a read command for controllers
        for controller_num in range(4):
            # In a real implementation, we would parse the PIF RAM command structure
            # This is a simplified version that just shows the concept
            
            # Write controller data to PIF RAM (simplified)
            if controller_num == 0:  # Only player 1 for now
                state = self.controllers[controller_num].get_state()
                for i, b in enumerate(state):
                    self.memory.pif_ram[0x01 + i] = b

# ========== Emulator Core ========== #
class EmulatorCore:
    """The core emulation engine that coordinates all components."""
    def __init__(self):
        self.memory = MemorySystem()
        self.cpu = CPU(self.memory)
        self.video_plugin = UltraHLEVideo(self.memory)
        self.audio_plugin = UltraHLEAudio(self.memory)
        self.input_plugin = InputPlugin(self.memory)
        
        self.rom_info = None
        self.running = False
        self.paused = False
        self.step_mode = False
        self.speed_factor = 1.0
        
        # Statistics
        self.fps = 0
        self.frame_time = 0
        self.total_frames = 0
        self.total_instructions = 0
        
        # Timing
        self.start_time = 0
        self.last_frame_time = 0
        self.frame_count = 0
        self.last_second_time = 0
        
        logger.info("Emulator Core initialized")
    
    def reset(self):
        """Reset the emulator to initial state."""
        self.cpu.reset()
        self.running = False
        self.paused = False
        self.step_mode = False
        
        self.fps = 0
        self.frame_time = 0
        self.total_frames = 0
        self.total_instructions = 0
        
        logger.info("Emulator Core reset")
    
    def load_rom(self, filename: str) -> ROMInfo:
        """Load an N64 ROM file."""
        try:
            with open(filename, "rb") as f:
                rom_data = f.read()
            
            # Get ROM info
            self.rom_info = ROMInfo.from_data(filename, rom_data)
            
            # Convert to native format if needed
            rom_data_native = bytearray(rom_data)
            convert_rom_to_native(rom_data_native, self.rom_info.format)
            
            # Load into memory
            self.memory.load_rom(rom_data_native)
            
            # Reset CPU and devices
            self.reset()
            
            logger.info(f"ROM loaded: {self.rom_info.internal_name} "
                      f"({len(rom_data_native)} bytes)")
            
            return self.rom_info
        
        except Exception as e:
            logger.error(f"Failed to load ROM: {e}")
            raise
    
    def start(self):
        """Start emulation."""
        if not self.running:
            self.running = True
            self.paused = False
            self.start_time = time.time()
            self.last_second_time = time.time()
            self.last_frame_time = time.time()
            logger.info("Emulation started")
    
    def stop(self):
        """Stop emulation."""
        if self.running:
            self.running = False
            logger.info("Emulation stopped")
    
    def pause(self):
        """Pause emulation."""
        self.paused = not self.paused
        logger.info(f"Emulation {'paused' if self.paused else 'resumed'}")
    
    def toggle_step_mode(self):
        """Toggle instruction stepping mode."""
        self.step_mode = not self.step_mode
        logger.info(f"Step mode {'enabled' if self.step_mode else 'disabled'}")
    
    def step_instruction(self):
        """Execute a single instruction in step mode."""
        if self.step_mode and self.running:
            instruction = self.cpu.step()
            logger.debug(f"Step: 0x{self.cpu.pc:08X} - {instruction}")
            return instruction
        return None
    
    def run_frame(self):
        """Run emulation for a single frame."""
        if not self.running or self.paused:
            return False
        
        frame_start_time = time.time()
        
        # Execute instructions until a frame is ready
        instructions_this_frame = 0
        max_instructions = 100000  # Safety limit
        
        while instructions_this_frame < max_instructions:
            # Execute CPU instruction
            self.cpu.step()
            instructions_this_frame += 1
            self.total_instructions += 1
            
            # Check for video frame completion
            if self.video_plugin.is_frame_ready():
                self.video_plugin.reset_frame_flag()
                self.total_frames += 1
                break
            
            # In step mode, only execute one instruction
            if self.step_mode:
                break
        
        # Process audio
        self.audio_plugin.process_audio_commands()
        
        # Update input state
        self.input_plugin.update_pif_ram()
        
        # Update timing statistics
        now = time.time()
        self.frame_time = now - frame_start_time
        self.frame_count += 1
        
        # Calculate FPS every second
        if now - self.last_second_time >= 1.0:
            self.fps = self.frame_count / (now - self.last_second_time)
            self.frame_count = 0
            self.last_second_time = now
        
        return True
    
    def get_stats(self) -> dict:
        """Return emulation statistics."""
        return {
            "fps": self.fps,
            "frame_time_ms": self.frame_time * 1000,
            "total_frames": self.total_frames,
            "total_instructions": self.total_instructions,
            "instructions_per_frame": self.total_instructions / max(1, self.total_frames),
            "running_time": time.time() - self.start_time if self.running else 0
        }

# ========== Emulator Thread ========== #
class EmulatorThread(threading.Thread):
    """Thread to run the emulation loop."""
    def __init__(self, emulator: EmulatorCore, update_callback=None):
        super().__init__()
        self.emulator = emulator
        self.update_callback = update_callback
        self._stop_event = threading.Event()
        self.daemon = True  # Thread will exit when main program exits
    
    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()
        self.emulator.stop()
    
    def run(self):
        """Main emulation loop."""
        logger.info("Emulator thread started")
        
        while not self._stop_event.is_set():
            # Run a frame of emulation
            if self.emulator.run_frame():
                # Call update callback if provided
                if self.update_callback:
                    self.update_callback()
            
            # Throttle to approximate real-time speed
            target_frame_time = 1.0 / 60.0  # Target 60 FPS
            elapsed = self.emulator.frame_time
            if elapsed < target_frame_time:
                time.sleep(target_frame_time - elapsed)
        
        logger.info("Emulator thread stopped")

# ========== Main Application Window ========== #
class MainWindow(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("UltraHLE-Style N64 Emulator (Python)")
        self.geometry("800x600")
        self.minsize(640, 480)
        
        # Set application icon
        # (In a real app, you'd include the icon file with your distribution)
        # self.iconphoto(True, tk.PhotoImage(file="n64_icon.png"))
        
        # Emulator components
        self.emulator = EmulatorCore()
        self.emulator_thread = None
        
        # UI setup
        self._create_menu()
        self._create_toolbar()
        self._create_display()
        self._create_statusbar()
        
        # Key bindings
        self._setup_key_bindings()
        
        # Update timer (for UI)
        self.after(16, self._update_ui)
        
        logger.info("Main window initialized")
    
    def _create_menu(self):
        """Create application menu."""
        menubar = tk.Menu(self)
        
        # File menu
        file_menu = tk.Menu(menubar, tearoff=False)
        file_menu.add_command(label="Load ROM...", command=self._open_rom, accelerator="Ctrl+O")
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self._on_exit, accelerator="Alt+F4")
        menubar.add_cascade(label="File", menu=file_menu)
        
        # Emulation menu
        emulation_menu = tk.Menu(menubar, tearoff=False)
        emulation_menu.add_command(label="Start", command=self._start_emulation, accelerator="F5")
        emulation_menu.add_command(label="Stop", command=self._stop_emulation, accelerator="F6")
        emulation_menu.add_command(label="Pause/Resume", command=self._pause_emulation, accelerator="F7")
        emulation_menu.add_separator()
        emulation_menu.add_command(label="Reset", command=self._reset_emulation, accelerator="F8")
        emulation_menu.add_separator()
        emulation_menu.add_command(label="Step Mode", command=self._toggle_step_mode, accelerator="F11")
        emulation_menu.add_command(label="Step Instruction", command=self._step_instruction, accelerator="F10")
        menubar.add_cascade(label="Emulation", menu=emulation_menu)
        
        # Options menu
        options_menu = tk.Menu(menubar, tearoff=False)
        options_menu.add_command(label="Settings...", command=self._show_settings)
        menubar.add_cascade(label="Options", menu=options_menu)
        
        # Debug menu
        debug_menu = tk.Menu(menubar, tearoff=False)
        debug_menu.add_command(label="Memory Viewer", command=self._show_memory_viewer)
        debug_menu.add_command(label="Debugger", command=self._show_debugger)
        debug_menu.add_command(label="CPU Registers", command=self._show_registers)
        menubar.add_cascade(label="Debug", menu=debug_menu)
        
        # Help menu
        help_menu = tk.Menu(menubar, tearoff=False)
        help_menu.add_command(label="About", command=self._show_about)
        menubar.add_cascade(label="Help", menu=help_menu)
        
        self.config(menu=menubar)
    
    def _create_toolbar(self):
        """Create toolbar with common actions."""
        toolbar_frame = ttk.Frame(self)
        toolbar_frame.pack(side=tk.TOP, fill=tk.X)
        
        # Load ROM button
        load_btn = ttk.Button(toolbar_frame, text="Load ROM", command=self._open_rom)
        load_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Start button
        start_btn = ttk.Button(toolbar_frame, text="Start", command=self._start_emulation)
        start_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Stop button
        stop_btn = ttk.Button(toolbar_frame, text="Stop", command=self._stop_emulation)
        stop_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Pause button
        pause_btn = ttk.Button(toolbar_frame, text="Pause", command=self._pause_emulation)
        pause_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Reset button
        reset_btn = ttk.Button(toolbar_frame, text="Reset", command=self._reset_emulation)
        reset_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Separator
        ttk.Separator(toolbar_frame, orient=tk.VERTICAL).pack(side=tk.LEFT, fill=tk.Y, padx=5, pady=2)
        
        # Step mode button
        step_mode_btn = ttk.Button(toolbar_frame, text="Step Mode", command=self._toggle_step_mode)
        step_mode_btn.pack(side=tk.LEFT, padx=2, pady=2)
        
        # Step instruction button
        step_btn = ttk.Button(toolbar_frame, text="Step", command=self._step_instruction)
        step_btn.pack(side=tk.LEFT, padx=2, pady=2)
    
    def _create_display(self):
        """Create the main display area."""
        # Main display frame
        self.display_frame = ttk.Frame(self)
        self.display_frame.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # Game display canvas
        self.canvas = tk.Canvas(self.display_frame, bg="black", width=640, height=480)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        
        # Display ROM info if loaded
        self.rom_info_frame = ttk.LabelFrame(self.display_frame, text="ROM Information")
        self.rom_info_frame.pack(side=tk.BOTTOM, fill=tk.X, pady=5)
        
        rom_info_grid = ttk.Frame(self.rom_info_frame)
        rom_info_grid.pack(fill=tk.X, padx=5, pady=5)
        
        # ROM info labels
        ttk.Label(rom_info_grid, text="Name:").grid(row=0, column=0, sticky=tk.W, padx=5)
        ttk.Label(rom_info_grid, text="Size:").grid(row=1, column=0, sticky=tk.W, padx=5)
        ttk.Label(rom_info_grid, text="Format:").grid(row=2, column=0, sticky=tk.W, padx=5)
        ttk.Label(rom_info_grid, text="Country:").grid(row=3, column=0, sticky=tk.W, padx=5)
        
        self.rom_name_var = tk.StringVar(value="No ROM loaded")
        self.rom_size_var = tk.StringVar(value="-")
        self.rom_format_var = tk.StringVar(value="-")
        self.rom_country_var = tk.StringVar(value="-")
        
        ttk.Label(rom_info_grid, textvariable=self.rom_name_var).grid(row=0, column=1, sticky=tk.W, padx=5)
        ttk.Label(rom_info_grid, textvariable=self.rom_size_var).grid(row=1, column=1, sticky=tk.W, padx=5)
        ttk.Label(rom_info_grid, textvariable=self.rom_format_var).grid(row=2, column=1, sticky=tk.W, padx=5)
        ttk.Label(rom_info_grid, textvariable=self.rom_country_var).grid(row=3, column=1, sticky=tk.W, padx=5)
    
    def _create_statusbar(self):
        """Create the status bar."""
        statusbar = ttk.Frame(self, relief=tk.SUNKEN, border=1)
        statusbar.pack(side=tk.BOTTOM, fill=tk.X)
        
        # Status message
        self.status_var = tk.StringVar(value="Ready")
        status_label = ttk.Label(statusbar, textvariable=self.status_var, anchor=tk.W)
        status_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)
        
        # FPS counter
        self.fps_var = tk.StringVar(value="0.0 FPS")
        fps_label = ttk.Label(statusbar, textvariable=self.fps_var)
        fps_label.pack(side=tk.RIGHT, padx=5)
        
        # Instructions counter
        self.instr_var = tk.StringVar(value="0 instrs")
        instr_label = ttk.Label(statusbar, textvariable=self.instr_var)
        instr_label.pack(side=tk.RIGHT, padx=5)
    
    def _setup_key_bindings(self):
        """Set up keyboard shortcuts and controller emulation."""
        # Menu shortcuts
        self.bind("<Control-o>", lambda e: self._open_rom())
        self.bind("<F5>", lambda e: self._start_emulation())
        self.bind("<F6>", lambda e: self._stop_emulation())
        self.bind("<F7>", lambda e: self._pause_emulation())
        self.bind("<F8>", lambda e: self._reset_emulation())
        self.bind("<F10>", lambda e: self._step_instruction())
        self.bind("<F11>", lambda e: self._toggle_step_mode())
        
        # Controller keys
        controller_keys = [
            'Up', 'Down', 'Left', 'Right',
            'z', 'x', 'c', 'Return',
            'a', 's', 'd', 'w',
            'q', 'e'
        ]
        
        for key in controller_keys:
            self.bind(f"<KeyPress-{key}>", lambda e, k=key: self._handle_key(k, True))
            self.bind(f"<KeyRelease-{key}>", lambda e, k=key: self._handle_key(k, False))
    
    def _handle_key(self, key: str, pressed: bool):
        """Handle keyboard input for controller emulation."""
        if self.emulator and self.emulator.input_plugin:
            self.emulator.input_plugin.process_key_event(key, pressed)
    
    def _open_rom(self):
        """Prompt user to select and load a ROM file."""
        filename = filedialog.askopenfilename(
            title="Open N64 ROM",
            filetypes=[
                ("N64 ROMs", "*.z64 *.v64 *.n64"),
                ("All Files", "*.*")
            ]
        )
        
        if filename:
            try:
                self.status_var.set(f"Loading ROM: {filename}")
                self.update_idletasks()
                
                # Stop any running emulation
                self._stop_emulation()
                
                # Load the ROM
                rom_info = self.emulator.load_rom(filename)
                
                # Update ROM info display
                self._update_rom_info(rom_info)
                
                self.status_var.set(f"ROM loaded: {rom_info.internal_name}")
            except Exception as e:
                messagebox.showerror("Error", f"Failed to load ROM: {str(e)}")
                self.status_var.set("Failed to load ROM")
    
    def _update_rom_info(self, rom_info: ROMInfo):
        """Update the ROM information display."""
        if rom_info:
            # Format ROM size in human-readable form
            size_str = f"{rom_info.size / (1024*1024):.2f} MB" if rom_info.size >= 1024*1024 else \
                      f"{rom_info.size / 1024:.2f} KB"
            
            # Get ROM format name
            format_names = {
                ROMFormat.Z64_BIG_ENDIAN: "Z64 (Big Endian)",
                ROMFormat.V64_BYTESWAPPED: "V64 (Byte Swapped)",
                ROMFormat.N64_LITTLE_ENDIAN: "N64 (Little Endian)",
                ROMFormat.UNKNOWN: "Unknown"
            }
            format_str = format_names.get(rom_info.format, "Unknown")
            
            # Get country code information
            country_codes = {
                'A': "Australia",
                'B': "Brazil",
                'C': "China",
                'D': "Germany",
                'E': "North America",
                'F': "France",
                'G': "Gateway 64 (NTSC)",
                'H': "Netherlands",
                'I': "Italy",
                'J': "Japan",
                'K': "Korea",
                'L': "Gateway 64 (PAL)",
                'N': "Canada",
                'P': "Europe",
                'S': "Spain",
                'U': "Australia",
                'W': "Scandinavia",
                'X': "Europe",
                'Y': "Europe"
            }
            country_str = country_codes.get(rom_info.country_code, "Unknown")
            
            # Update display
            self.rom_name_var.set(rom_info.internal_name)
            self.rom_size_var.set(size_str)
            self.rom_format_var.set(format_str)
            self.rom_country_var.set(f"{rom_info.country_code} - {country_str}")
        else:
            # Clear display if no ROM
            self.rom_name_var.set("No ROM loaded")
            self.rom_size_var.set("-")
            self.rom_format_var.set("-")
            self.rom_country_var.set("-")
    
    def _start_emulation(self):
        """Start the emulation."""
        if not self.emulator.rom_info:
            messagebox.showinfo("No ROM", "Please load a ROM first.")
            return
        
        if self.emulator_thread and self.emulator_thread.is_alive():
            # Already running
            return
        
        # Start the emulator
        self.emulator.start()
        
        # Create and start thread
        self.emulator_thread = EmulatorThread(self.emulator, self._on_frame_update)
        self.emulator_thread.start()
        
        self.status_var.set("Emulation started")
    
    def _stop_emulation(self):
        """Stop the emulation."""
        if self.emulator_thread and self.emulator_thread.is_alive():
            self.emulator_thread.stop()
            self.emulator_thread.join(timeout=1.0)
            self.emulator_thread = None
        
        self.emulator.stop()
        self.status_var.set("Emulation stopped")
    
    def _pause_emulation(self):
        """Pause or resume emulation."""
        if self.emulator:
            self.emulator.pause()
            status = "paused" if self.emulator.paused else "resumed"
            self.status_var.set(f"Emulation {status}")
    
    def _reset_emulation(self):
        """Reset the emulation."""
        self._stop_emulation()
        self.emulator.reset()
        self.status_var.set("Emulation reset")
    
    def _toggle_step_mode(self):
        """Toggle instruction stepping mode."""
        if self.emulator:
            self.emulator.toggle_step_mode()
            status = "enabled" if self.emulator.step_mode else "disabled"
            self.status_var.set(f"Step mode {status}")
    
    def _step_instruction(self):
        """Execute a single instruction in step mode."""
        if self.emulator.step_mode and self.emulator.running:
            instruction = self.emulator.step_instruction()
            if instruction:
                self.status_var.set(f"Step: 0x{self.emulator.cpu.pc:08X} - {instruction}")
                # Update display
                self._update_ui()
    
    def _on_frame_update(self):
        """Called when a new frame is rendered."""
        # This is called from the emulator thread
        # We don't do UI updates directly from here
        pass
    
    def _update_ui(self):
        """Update UI components with current emulator state."""
        if self.emulator:
            # Update statistics
            stats = self.emulator.get_stats()
            self.fps_var.set(f"{stats['fps']:.1f} FPS")
            self.instr_var.set(f"{stats['total_instructions']} instrs")
            
            # Update game display
            if self.emulator.video_plugin and self.emulator.running:
                try:
                    # Get the current frame as a PIL Image
                    frame = self.emulator.video_plugin.get_framebuffer_image()
                    if frame:
                        # Convert to Tkinter PhotoImage
                        self.photo = ImageTk.PhotoImage(frame)
                        # Update canvas
                        self.canvas.delete("all")
                        self.canvas.create_image(
                            self.canvas.winfo_width() // 2,
                            self.canvas.winfo_height() // 2,
                            image=self.photo
                        )
                except Exception as e:
                    logger.error(f"Error updating display: {e}")
        
        # Schedule next update
        self.after(16, self._update_ui)  # ~60 FPS
    
    def _show_settings(self):
        """Show settings dialog."""
        settings_window = tk.Toplevel(self)
        settings_window.title("Settings")
        settings_window.geometry("400x300")
        settings_window.transient(self)
        settings_window.grab_set()
        
        # Settings content
        ttk.Label(settings_window, text="Settings", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Controller settings
        controller_frame = ttk.LabelFrame(settings_window, text="Controller Settings")
        controller_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Plugin settings
        plugins_frame = ttk.LabelFrame(settings_window, text="Plugins")
        plugins_frame.pack(fill=tk.X, padx=10, pady=5)
        
        # Video plugin
        ttk.Label(plugins_frame, text="Video Plugin:").grid(row=0, column=0, sticky=tk.W, padx=5, pady=5)
        video_plugin = ttk.Combobox(plugins_frame, values=["UltraHLE Video (Default)"])
        video_plugin.current(0)
        video_plugin.grid(row=0, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Audio plugin
        ttk.Label(plugins_frame, text="Audio Plugin:").grid(row=1, column=0, sticky=tk.W, padx=5, pady=5)
        audio_plugin = ttk.Combobox(plugins_frame, values=["UltraHLE Audio (Default)"])
        audio_plugin.current(0)
        audio_plugin.grid(row=1, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Input plugin
        ttk.Label(plugins_frame, text="Input Plugin:").grid(row=2, column=0, sticky=tk.W, padx=5, pady=5)
        input_plugin = ttk.Combobox(plugins_frame, values=["Keyboard Input (Default)"])
        input_plugin.current(0)
        input_plugin.grid(row=2, column=1, sticky=tk.W, padx=5, pady=5)
        
        # Buttons
        btn_frame = ttk.Frame(settings_window)
        btn_frame.pack(side=tk.BOTTOM, fill=tk.X, padx=10, pady=10)
        
        ttk.Button(btn_frame, text="OK", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)
        ttk.Button(btn_frame, text="Cancel", command=settings_window.destroy).pack(side=tk.RIGHT, padx=5)
    
    def _show_memory_viewer(self):
        """Show memory viewer window."""
        memory_window = tk.Toplevel(self)
        memory_window.title("Memory Viewer")
        memory_window.geometry("600x400")
        memory_window.transient(self)
        
        # Memory viewer content
        ttk.Label(memory_window, text="Memory Viewer", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Memory region selection
        region_frame = ttk.Frame(memory_window)
        region_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(region_frame, text="Region:").pack(side=tk.LEFT, padx=5)
        region_combo = ttk.Combobox(region_frame, values=[
            "RDRAM (0x00000000)",
            "RSP DMEM (0x04000000)",
            "RSP IMEM (0x04001000)",
            "Cartridge ROM (0x10000000)",
            "PIF ROM (0x1FC00000)"
        ])
        region_combo.current(0)
        region_combo.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        # Address entry
        addr_frame = ttk.Frame(memory_window)
        addr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(addr_frame, text="Address:").pack(side=tk.LEFT, padx=5)
        addr_entry = ttk.Entry(addr_frame)
        addr_entry.insert(0, "00000000")
        addr_entry.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        ttk.Button(addr_frame, text="Go").pack(side=tk.LEFT, padx=5)
        
        # Memory hex view
        hex_frame = ttk.Frame(memory_window)
        hex_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(hex_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        # Text widget for hex display
        hex_text = tk.Text(hex_frame, yscrollcommand=scrollbar.set, font=("Courier", 10))
        hex_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=hex_text.yview)
        
        # Insert sample data
        for i in range(16):
            addr = i * 16
            hex_text.insert(tk.END, f"{addr:08X}:  ")
            for j in range(16):
                hex_text.insert(tk.END, f"{(i*16 + j) % 256:02X} ")
            hex_text.insert(tk.END, "  ")
            for j in range(16):
                char = chr((i*16 + j) % 256)
                if 32 <= (i*16 + j) % 256 <= 126:
                    hex_text.insert(tk.END, char)
                else:
                    hex_text.insert(tk.END, ".")
            hex_text.insert(tk.END, "\n")
        
        hex_text.config(state=tk.DISABLED)
    
    def _show_debugger(self):
        """Show debugger window."""
        debugger_window = tk.Toplevel(self)
        debugger_window.title("Debugger")
        debugger_window.geometry("600x500")
        debugger_window.transient(self)
        
        # Debugger content
        ttk.Label(debugger_window, text="Debugger", font=("Arial", 14, "bold")).pack(pady=10)
        
        # Address entry
        addr_frame = ttk.Frame(debugger_window)
        addr_frame.pack(fill=tk.X, padx=10, pady=5)
        
        ttk.Label(addr_frame, text="PC:").pack(side=tk.LEFT, padx=5)
        pc_entry = ttk.Entry(addr_frame, width=10)
        pc_entry.insert(0, f"{self.emulator.cpu.pc:08X}" if self.emulator else "00000000")
        pc_entry.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(addr_frame, text="Go").pack(side=tk.LEFT, padx=5)
        ttk.Button(addr_frame, text="Step").pack(side=tk.LEFT, padx=5)
        ttk.Button(addr_frame, text="Step Over").pack(side=tk.LEFT, padx=5)
        ttk.Button(addr_frame, text="Run").pack(side=tk.LEFT, padx=5)
        
        # Disassembly view
        disasm_frame = ttk.LabelFrame(debugger_window, text="Disassembly")
        disasm_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)
        
        # Scrollbar
        scrollbar = ttk.Scrollbar(disasm_
