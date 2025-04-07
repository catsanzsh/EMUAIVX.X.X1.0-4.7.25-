#include <QApplication>
#include <QMainWindow>
#include <QPushButton>
#include <QVBoxLayout>
#include <QFileDialog>
#include <QLabel>
#include <QStatusBar>
#include <QString>
#include <QMenuBar>
#include <QMenu>
#include <QAction>
#include <QThread>
#include <QMutex>
#include <QWaitCondition>
#include <QTimer>
#include <iostream>
#include <fstream>
#include <atomic>
#include <algorithm>

// ========== N64 CPU (MIPS R4300) Definitions ========== //
#define MEMORY_SIZE  0x800000 // 8 MB typical for N64
#define R4300_REGS   32

// Basic CPU structure
struct CPU {
    uint32_t pc = 0;
    uint32_t regs[R4300_REGS] = {0};
    bool running = false;
};

// Global CPU state and memory
static CPU cpu;
static uint8_t *gMem = nullptr;  // Use a byte array to store ROM + RDRAM

// Some atomic flags for thread-safe control
static std::atomic<bool> gEmulationRunning{false};
static std::atomic<bool> gStopRequested{false};

static QMutex gMemoryMutex;
static QWaitCondition gMemoryCond;

// ========== CPU / Memory / ROM Handling ========== //
void init_r4300()
{
    cpu.pc = 0x1000; // Just an example reset vector
    std::fill(std::begin(cpu.regs), std::end(cpu.regs), 0);
    cpu.running = true;
}

int load_rom(const char *filename)
{
    std::ifstream romfile(filename, std::ios::binary);
    if(!romfile.is_open()) {
        std::cerr << "Error: Unable to open ROM file." << std::endl;
        return -1;
    }

    // Lock memory while loading
    QMutexLocker locker(&gMemoryMutex);

    // For a typical N64 ROM, sizes can range up to 64MB.
    // Here we assume gMem is large enough to store max ROM size,
    // though this is just for demonstration.
    romfile.seekg(0, std::ios::end);
    std::streamsize romSize = romfile.tellg();
    romfile.seekg(0, std::ios::beg);

    if (romSize > MEMORY_SIZE) {
        std::cerr << "Warning: ROM size exceeds allocated memory. Attempting partial load." << std::endl;
        romSize = MEMORY_SIZE;
    }

    romfile.read(reinterpret_cast<char*>(gMem), romSize);
    romfile.close();

    // Usually you'd do checks or decode the ROM header to set initial PC, etc.
    // For demonstration, we just set pc to a test address after loading.
    cpu.pc = 0x1000;
    return 0;
}

/**
 * Example instruction decoding
 *
 * This is far from complete. The real MIPS R4300i core includes 
 * many more instructions, exceptions, TLB management, etc.
 */
void run_r4300_instruction()
{
    // Very simplified fetch
    // We assume pc is word-aligned, but real hardware checks for alignment.
    uint32_t addr = cpu.pc & 0x1FFFFFC; // Just ensure we don't exceed memory
    uint32_t instruction = 0;

    {
        QMutexLocker locker(&gMemoryMutex);
        if (addr + 3 < MEMORY_SIZE) {
            instruction = (gMem[addr + 0] << 24) |
                          (gMem[addr + 1] << 16) |
                          (gMem[addr + 2] <<  8) |
                           gMem[addr + 3];
        } else {
            // If PC is out of range, just stop
            cpu.running = false;
            return;
        }
    }

    cpu.pc += 4;  // Next instruction

    uint8_t opcode = (instruction >> 26) & 0x3F;
    switch (opcode)
    {
        case 0x08: // ADDI (simplified)
        {
            uint8_t rt = (instruction >> 16) & 0x1F;
            uint8_t rs = (instruction >> 21) & 0x1F;
            int16_t imm = (int16_t)(instruction & 0xFFFF);
            cpu.regs[rt] = cpu.regs[rs] + imm;
            break;
        }
        case 0x02: // J (jump)
        {
            uint32_t target = instruction & 0x03FFFFFF;
            // Basic jump handling
            cpu.pc = (cpu.pc & 0xF0000000) | (target << 2);
            break;
        }
        // Add more instructions here...
        default:
        {
            // 0x00 can be a NOP or an R-type instruction that we have to decode further
            // For demonstration, treat unknown instructions as NOP
            if (instruction == 0x00000000) {
                // Consider it a NOP
            }
            break;
        }
    }

    // If pc goes out of memory range, stop
    if (cpu.pc >= MEMORY_SIZE) {
        cpu.running = false;
    }
}

bool cpu_running() {
    return cpu.running;
}

// ========== Plugin Stubs (Video, Audio, Input) ========== //
class VideoPlugin {
public:
    virtual ~VideoPlugin() {}
    virtual void initialize() { std::cout << "[Video] Initialized plugin.\n"; }
    virtual void renderFrame() { /* Normally render graphics */ }
};

class LegacyVideoPlugin : public VideoPlugin {
public:
    void initialize() override {
        std::cout << "[LegacyVideo] Using old-school graphics plugin.\n";
    }
    void renderFrame() override {
        // Place logic for drawing a frame using old, fixed pipeline, etc.
    }
};

class AudioPlugin {
public:
    virtual ~AudioPlugin() {}
    virtual void initialize() { std::cout << "[Audio] Initialized plugin.\n"; }
    virtual void processAudio() {}
};

class InputPlugin {
public:
    virtual ~InputPlugin() {}
    virtual void initialize() { std::cout << "[Input] Initialized plugin.\n"; }
    virtual void pollControls() {}
};

// ========== Threads for CPU & Rendering ========== //
class CPUThread : public QThread {
    Q_OBJECT
public:
    explicit CPUThread(QObject *parent = nullptr) : QThread(parent) {}
protected:
    void run() override {
        // Main CPU emulation loop
        while(!gStopRequested.load()) {
            if (cpu_running() && gEmulationRunning.load()) {
                run_r4300_instruction();
            } else {
                msleep(1); // Idle
            }
        }
    }
};

class VideoThread : public QThread {
    Q_OBJECT
public:
    VideoThread(VideoPlugin *plugin, QObject *parent = nullptr)
        : QThread(parent), m_videoPlugin(plugin) {}

protected:
    void run() override {
        while(!gStopRequested.load()) {
            if (cpu_running() && gEmulationRunning.load()) {
                // Render a frame (placeholder)
                m_videoPlugin->renderFrame();
            }
            msleep(16); // ~60fps
        }
    }
private:
    VideoPlugin *m_videoPlugin;
};

// ========== MainWindow With Legacy GUI Style Menu ========== //
class MainWindow : public QMainWindow {
    Q_OBJECT
public:
    MainWindow(QWidget *parent = nullptr);
    ~MainWindow();

private slots:
    void loadROM();
    void startEmulation();
    void stopEmulation();
    void configurePlugins();
    void aboutDialog();

private:
    void runEmulator();
    void stopEmulatorCore();

    QPushButton *loadButton;
    QPushButton *startButton;
    QPushButton *stopButton;
    QLabel      *statusDisplay;
    QString      loadedROM;

    // Plugins
    VideoPlugin  *videoPlugin;
    AudioPlugin  *audioPlugin;
    InputPlugin  *inputPlugin;

    // Threads
    CPUThread   *cpuThread;
    VideoThread *videoThread;
};

MainWindow::MainWindow(QWidget *parent) 
    : QMainWindow(parent),
      videoPlugin(nullptr),
      audioPlugin(nullptr),
      inputPlugin(nullptr),
      cpuThread(nullptr),
      videoThread(nullptr)
{
    setWindowTitle("Net64_Mupen (Project64 1.6 Legacy Style)");
    setFixedSize(800, 600);

    // Prepare menu bar & actions (legacy GUI style)
    QMenu *fileMenu = menuBar()->addMenu("&File");
    QAction *loadAction = new QAction("Load ROM", this);
    QAction *exitAction = new QAction("Exit", this);
    connect(loadAction, &QAction::triggered, this, &MainWindow::loadROM);
    connect(exitAction, &QAction::triggered, this, &MainWindow::close);
    fileMenu->addAction(loadAction);
    fileMenu->addSeparator();
    fileMenu->addAction(exitAction);

    QMenu *emulationMenu = menuBar()->addMenu("&Emulation");
    QAction *startAction = new QAction("Start", this);
    QAction *stopAction = new QAction("Stop", this);
    connect(startAction, &QAction::triggered, this, &MainWindow::startEmulation);
    connect(stopAction, &QAction::triggered, this, &MainWindow::stopEmulation);
    emulationMenu->addAction(startAction);
    emulationMenu->addAction(stopAction);

    QMenu *configMenu = menuBar()->addMenu("&Options");
    QAction *pluginCfg = new QAction("Configure Plugins...", this);
    connect(pluginCfg, &QAction::triggered, this, &MainWindow::configurePlugins);
    configMenu->addAction(pluginCfg);

    QMenu *helpMenu = menuBar()->addMenu("&Help");
    QAction *aboutAction = new QAction("About Net64_Mupen...", this);
    connect(aboutAction, &QAction::triggered, this, &MainWindow::aboutDialog);
    helpMenu->addAction(aboutAction);

    // Central widget
    QWidget *centralWidget = new QWidget(this);
    QVBoxLayout *layout = new QVBoxLayout(centralWidget);

    loadButton = new QPushButton("Load ROM", this);
    connect(loadButton, &QPushButton::clicked, this, &MainWindow::loadROM);

    startButton = new QPushButton("Start Emulation", this);
    connect(startButton, &QPushButton::clicked, this, &MainWindow::startEmulation);

    stopButton = new QPushButton("Stop Emulation", this);
    connect(stopButton, &QPushButton::clicked, this, &MainWindow::stopEmulation);

    statusDisplay = new QLabel("Status: Waiting", this);

    layout->addWidget(loadButton);
    layout->addWidget(startButton);
    layout->addWidget(stopButton);
    layout->addWidget(statusDisplay);

    setCentralWidget(centralWidget);
}

MainWindow::~MainWindow()
{
    stopEmulatorCore(); // Ensure threads are stopped
    delete[] gMem;
}

void MainWindow::loadROM()
{
    loadedROM = QFileDialog::getOpenFileName(this, "Open ROM File", "", "ROM Files (*.z64 *.n64 *.v64);;All Files (*)");
    if (!loadedROM.isEmpty()) {
        statusDisplay->setText("ROM Loaded: " + loadedROM);
    }
}

void MainWindow::startEmulation()
{
    if (loadedROM.isEmpty()) {
        statusDisplay->setText("No ROM loaded.");
        return;
    }

    statusDisplay->setText("Starting Emulation...");

    // Cleanup old threads if any
    stopEmulatorCore();

    // Initialize memory if needed
    if (!gMem) {
        gMem = new uint8_t[MEMORY_SIZE];
        std::fill_n(gMem, MEMORY_SIZE, 0);
    }

    init_r4300();
    if (load_rom(loadedROM.toStdString().c_str()) != 0) {
        statusDisplay->setText("Failed to load ROM.");
        return;
    }

    // Plugins
    if (!videoPlugin) {
        // Example: use a “legacy” video plugin
        videoPlugin = new LegacyVideoPlugin();
        videoPlugin->initialize();
    }
    if (!audioPlugin) {
        audioPlugin = new AudioPlugin();
        audioPlugin->initialize();
    }
    if (!inputPlugin) {
        inputPlugin = new InputPlugin();
        inputPlugin->initialize();
    }

    // Create threads
    gStopRequested.store(false);
    gEmulationRunning.store(true);

    cpuThread = new CPUThread(this);
    videoThread = new VideoThread(videoPlugin, this);

    cpuThread->start(QThread::HighPriority);
    videoThread->start(QThread::NormalPriority);

    statusDisplay->setText("Emulation Running.");
}

void MainWindow::stopEmulation()
{
    statusDisplay->setText("Stopping Emulation...");
    stopEmulatorCore();
    statusDisplay->setText("Emulation Stopped.");
}

void MainWindow::configurePlugins()
{
    // A real emulator would open a config dialog to pick from multiple plugin DLLs, etc.
    // For demonstration, we just show a message.
    statusBar()->showMessage("Plugin configuration placeholder...");
}

void MainWindow::aboutDialog()
{
    statusBar()->showMessage("Net64_Mupen: A hypothetical N64 emulator with PJ64 1.6 style UI");
}

void MainWindow::runEmulator()
{
    // (Unused in this example – Emulation is done in threads.)
}

void MainWindow::stopEmulatorCore()
{
    if (cpuThread) {
        gStopRequested.store(true);
        cpu.running = false;
        gEmulationRunning.store(false);

        cpuThread->wait();
        videoThread->wait();

        delete cpuThread;
        cpuThread = nullptr;
        delete videoThread;
        videoThread = nullptr;
    }

    // Clean up plugins
    if (videoPlugin) {
        delete videoPlugin;
        videoPlugin = nullptr;
    }
    if (audioPlugin) {
        delete audioPlugin;
        audioPlugin = nullptr;
    }
    if (inputPlugin) {
        delete inputPlugin;
        inputPlugin = nullptr;
    }

    gStopRequested.store(false);
}

int main(int argc, char *argv[])
{
    QApplication app(argc, argv);
    MainWindow w;
    w.show();
    return app.exec();
}

#include "Net64_Mupen.moc"
