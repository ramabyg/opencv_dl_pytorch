#include <torch/torch.h>
#include <iostream>

using namespace std;

int main() {
  try {
    cout << "Starting program..." << endl;
    cout.flush();

    torch::Tensor tensor = torch::randn({3,3});
    cout << "The random matrix is:" << endl << tensor << endl;
    cout.flush();

    //Initialize the device to CPU
    torch::DeviceType device = torch::kCPU;
    //If CUDA is available,run on GPU
    if (torch::cuda::is_available())
        device = torch::kCUDA;
    cout << "Running on: "
              << (device == torch::kCUDA ? "GPU" : "CPU") << endl;
    cout.flush();

    cout << "Program completed successfully!" << endl;
    cout.flush();

    return 0;
  }
  catch (const exception& e) {
    cerr << "Error: " << e.what() << endl;
    cerr.flush();
    return 1;
  }
  catch (...) {
    cerr << "Unknown error occurred" << endl;
    cerr.flush();
    return 1;
  }
}
