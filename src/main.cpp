#include "cv40.hpp"
#include <charconv>
#ifdef _WIN32
#include <windows.h>

#include <shellapi.h>
#endif

namespace {
using namespace cv40;

enum class Action { Run, AllSafe, List, SelfTest, Help };

struct Options {
    Context context;
    std::optional<Action> action;
    std::string exampleId;
    int seed = 42;
};

void printHelp() {
    std::cout << "cv40 --list | --run ID [--headless] [--data CODE_DIR] [--models DIR]\n"
                 "     [--output DIR] [--input 90,2] [--seed 42]\n"
                 "     [--camera 0 | --frames DIR] [--max-frames N]\n"
                 "cv40 --all-safe | --self-test\nC++17 / OpenCV 4 / dlib\n";
}

int nonnegativeInteger(const std::string &text, const std::string &option) {
    int value = 0;
    const auto result = std::from_chars(text.data(), text.data() + text.size(), value);
    require(result.ec == std::errc() && result.ptr == text.data() + text.size() && value >= 0,
            option + " expects a nonnegative integer: " + text);
    return value;
}

std::vector<double> numericInputs(const std::string &text) {
    require(!text.empty() && text.back() != ',', "--input expects comma-separated finite numbers");
    std::vector<double> values;
    std::istringstream stream(text);
    std::string token;
    while (std::getline(stream, token, ',')) {
        std::istringstream number(token);
        double value = 0;
        require(bool(number >> value) && std::isfinite(value), "Invalid --input value: " + token);
        number >> std::ws;
        require(number.eof(), "Invalid --input value: " + token);
        values.push_back(value);
    }
    return values;
}

Options parseOptions(const std::vector<std::string> &arguments) {
    Options options;
    options.context.root = fs::u8path(CV40_DEFAULT_ROOT);
    options.context.output = fs::current_path() / "output";
    bool cameraSpecified = false;
    auto selectAction = [&](Action action) {
        require(!options.action, "Select only one of --run, --all-safe, --list, --self-test or --help");
        options.action = action;
    };
    for (std::size_t index = 1; index < arguments.size(); ++index) {
        const std::string &argument = arguments[index];
        auto value = [&]() {
            require(index + 1 < arguments.size(), "Missing value for " + argument);
            return arguments[++index];
        };
        if (argument == "--list") {
            selectAction(Action::List);
        } else if (argument == "--self-test") {
            selectAction(Action::SelfTest);
        } else if (argument == "--help" || argument == "-h") {
            selectAction(Action::Help);
        } else if (argument == "--data") {
            options.context.root = fs::u8path(value());
        } else if (argument == "--models") {
            options.context.models = fs::u8path(value());
        } else if (argument == "--output") {
            options.context.output = fs::u8path(value());
        } else if (argument == "--frames") {
            options.context.frames = fs::u8path(value());
        } else if (argument == "--headless") {
            options.context.headless = true;
        } else if (argument == "--all-safe") {
            selectAction(Action::AllSafe);
            options.context.headless = true;
        } else if (argument == "--camera") {
            cameraSpecified = true;
            options.context.camera = nonnegativeInteger(value(), argument);
        } else if (argument == "--max-frames") {
            options.context.maxFrames = nonnegativeInteger(value(), argument);
            require(options.context.maxFrames > 0, "--max-frames must be greater than zero");
        } else if (argument == "--seed") {
            options.seed = nonnegativeInteger(value(), argument);
        } else if (argument == "--input") {
            options.context.input = numericInputs(value());
        } else if (argument == "--run" || (!argument.empty() && argument[0] != '-')) {
            selectAction(Action::Run);
            options.exampleId = argument == "--run" ? value() : argument;
        } else {
            throw std::runtime_error("Unknown option: " + argument);
        }
    }
    require(options.action.has_value(), "Select an example with --run ID; use --list or --help");
    require(!cameraSpecified || options.context.frames.empty(), "Use either --camera or --frames");
    return options;
}

int runExamples(Options &options) {
    Context &context = options.context;
    const bool allSafe = *options.action == Action::AllSafe;
    int passed = 0, failed = 0, skipped = 0;
    bool found = false;
    for (const Example &example : examples) {
        if (!allSafe && options.exampleId != example.id) {
            continue;
        }
        if (allSafe && (example.camera || example.model)) {
            ++skipped;
            continue;
        }
        found = true;
        context.example = &example;
        setRNGSeed(options.seed);
        try {
            std::cout << "RUN " << example.id << " " << example.source << '\n';
            run(context);
            ++passed;
            std::cout << "PASS " << example.id << '\n';
        } catch (const std::exception &error) {
            ++failed;
            std::cerr << "FAIL " << example.id << ": " << error.what() << '\n';
        }
        if (!context.headless && !example.camera && context.imageNumber > 0) {
            waitKey(0);
        }
        destroyAllWindows();
    }
    require(found, "Unknown example ID: " + options.exampleId);
    std::cout << "passed=" << passed << " failed=" << failed << " skipped=" << skipped << '\n';
    return failed ? 1 : 0;
}

std::vector<std::string> utf8Arguments(int argc, char **argv) {
    std::vector<std::string> arguments;
#ifdef _WIN32
    SetConsoleOutputCP(CP_UTF8);
    SetConsoleCP(CP_UTF8);
    wchar_t **wideArguments = CommandLineToArgvW(GetCommandLineW(), &argc);
    require(wideArguments != nullptr, "Cannot read command line");
    for (int index = 0; index < argc; ++index) {
        arguments.push_back(fs::path(wideArguments[index]).u8string());
    }
    LocalFree(wideArguments);
    (void)argv;
#else
    arguments.assign(argv, argv + argc);
#endif
    return arguments;
}
} // namespace

int main(int argc, char **argv) {
    try {
        Options options = parseOptions(utf8Arguments(argc, argv));
        switch (*options.action) {
        case Action::Help:
            printHelp();
            return 0;
        case Action::List:
            for (const auto &example : cv40::examples) {
                std::cout << example.id << '\t' << example.source << (example.camera ? " [camera]" : "")
                          << (example.model ? " [model]" : "") << '\n';
            }
            return 0;
        case Action::SelfTest:
            cv40::selfTest();
            return 0;
        default:
            return runExamples(options);
        }
    } catch (const std::exception &error) {
        std::cerr << error.what() << '\n';
        return 1;
    }
}
