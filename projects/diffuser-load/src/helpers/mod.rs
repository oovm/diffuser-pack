use candle_core::Device;
use candle_core::utils::{cuda_is_available, metal_is_available};

pub fn detect_device() -> candle_core::Result<Device> {
    if cuda_is_available() {
        Ok(Device::new_cuda(0)?)
    }
    else if metal_is_available() {
        Ok(Device::new_metal(0)?)
    }
    else {
        #[cfg(all(target_os = "macos", target_arch = "aarch64"))]
        {
            println!("Running on CPU, to run on GPU(metal), build this example with `--features metal`");
        }
        #[cfg(not(all(target_os = "macos", target_arch = "aarch64")))]
        {
            println!("Running on CPU, to run on GPU, build this example with `--features cuda`");
        }
        Ok(Device::Cpu)
    }
}