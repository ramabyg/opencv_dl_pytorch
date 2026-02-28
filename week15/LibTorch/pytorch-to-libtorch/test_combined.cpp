#include <torch/torch.h>
#include <torch/script.h>
#include <iostream>
#include <vector>

using namespace std;

int main() {
    try {
        cout << "Step 1: Basic tensor test..." << endl;
        torch::Tensor tensor = torch::randn({3,3});
        cout << "The random matrix is:" << endl << tensor << endl;

        cout << "Step 2: Loading model..." << endl;
        cout.flush();
        
        torch::jit::script::Module model = torch::jit::load("./LeNet.pt");
        
        cout << "Step 3: Model loaded successfully!" << endl;
        cout.flush();

        auto input = torch::randn({1, 1, 28, 28});
        cout << "Step 4: Created input" << endl;
        cout.flush();

        vector<torch::jit::IValue> jit_input;
        jit_input.push_back(input);
        
        auto output = model.forward(jit_input).toTensor();
        cout << "Step 5: Output size: " << output.sizes() << endl;

        //Initialize the device to CPU
        torch::DeviceType device = torch::kCPU;
        //If CUDA is available,run on GPU
        if (torch::cuda::is_available())
            device = torch::kCUDA;
        cout << "Running on: "
            << (device == torch::kCUDA ? "GPU" : "CPU") << endl;

        cout << "SUCCESS!" << endl;
        cout.flush();

        return 0;
    }
    catch (const exception& e) {
        cerr << "Error: " << e.what() << endl;
        cerr.flush();
        return 1;
    }
}
