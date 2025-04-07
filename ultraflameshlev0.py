# Incomplete continuation of the _show_debugger method from MainWindow
        scrollbar = ttk.Scrollbar(disasm_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)

        # Text widget for disassembly
        disasm_text = tk.Text(disasm_frame, yscrollcommand=scrollbar.set, font=("Courier", 10))
        disasm_text.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scrollbar.config(command=disasm_text.yview)

        # Populate with mock disassembly if emulator is running
        if self.emulator and self.emulator.cpu:
            pc = self.emulator.cpu.pc
            for i in range(32):
                opcode = self.emulator.memory.read32(pc + i * 4)
                instr = Instruction(opcode)
                disasm_text.insert(tk.END, f"{pc + i*4:08X}: {instr}\n")

        disasm_text.config(state=tk.DISABLED)

    def _show_registers(self):
        """Show CPU register window."""
        reg_window = tk.Toplevel(self)
        reg_window.title("CPU Registers")
        reg_window.geometry("400x500")
        reg_window.transient(self)

        ttk.Label(reg_window, text="CPU Registers", font=("Arial", 14, "bold")).pack(pady=10)

        reg_frame = ttk.Frame(reg_window)
        reg_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        if self.emulator and self.emulator.cpu:
            for i in range(32):
                ttk.Label(reg_frame, text=f"R{i:02}: {self.emulator.cpu.registers[i]:08X}").grid(row=i, column=0, sticky=tk.W, padx=5)

            ttk.Label(reg_frame, text=f"PC : {self.emulator.cpu.pc:08X}").grid(row=33, column=0, sticky=tk.W, padx=5)
            ttk.Label(reg_frame, text=f"HI : {self.emulator.cpu.hi:08X}").grid(row=34, column=0, sticky=tk.W, padx=5)
            ttk.Label(reg_frame, text=f"LO : {self.emulator.cpu.lo:08X}").grid(row=35, column=0, sticky=tk.W, padx=5)

    def _show_about(self):
        """Show about dialog."""
        messagebox.showinfo(
            "About UltraHLE Python",
            "UltraHLE-Style N64 Emulator\nDeveloped in Python with ❤️ by Flames-sama"
        )
