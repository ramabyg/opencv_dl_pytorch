#include <ostream>
#include<torch/script.h>
#include<iostream>
#include<vector>
#include <exception>

int main()
{
try {
std::cout << "Step 1: Program started" << std::endl;
std::cout.flush();

torch::DeviceType device = torch::kCPU;
std::cout << "Step 2: Device set to CPU" << std::endl;
std::cout.flush();

std::cout << "Step 3: Loading model..." << std::endl;
std::cout.flush();

torch::jit::script::Module model = torch::jit::load("./LeNet.pt", device);

std::cout << "Step 4: Model loaded successfully!" << std::endl;
std::cout.flush();

std::cout << "Step 5: Creating input tensor..." << std::endl;
auto input = torch::randn({1, 1, 28, 28}, device);
std::cout << "Step 6: Input created" << std::endl;
std::cout.flush();

std::vector<torch::jit::IValue> jit_input;
jit_input.push_back(input);

std::cout << "Step 7: Running inference..." << std::endl;
auto output = model.forward(jit_input).toTensor();
std::cout << "Step 8: Output size is " << output.sizes() << std::endl;
std::cout << "SUCCESS!" << std::endl;
std::cout.flush();

return 0;
}
catch (const std::exception& e) {
std::cerr << "Exception: " << e.what() << std::endl;
std::cerr.flush();
return 1;
}
catch (...) {
std::cerr << "UNKNOWN EXCEPTION!" << std::endl;
std::cerr.flush();
return 2;
}
}
