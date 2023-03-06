mod errors;

use std::io::Read;
use safetensors::tensor::{Dtype, SafeTensors, TensorView};

pub use errors::{Error, Result};

mod load_pickle;
pub use crate::load_pickle::load_pth;


#[test]
fn load_safe_tensors() {
    let mut file = std::fs::File::open("schoolAnime_schoolAnime.safetensors").unwrap();
    let mut buffer = Vec::new();
    file.read_to_end(&mut buffer).unwrap();
    let meta = SafeTensors::read_metadata(&buffer).unwrap();
    // let body = SafeTensors::deserialize(&buffer).unwrap();
    // for (name, tensor) in body.tensors() {
    //
    // }
    // TensorView::new();



    println!("{:#?}", meta);
    // println!("{:#?}", tnesor.names());
}

#[test]
fn test2() {
    let cm = tch::CModule::load("pureerosface_v1.pt").unwrap();

    println!("{:#?}", cm.named_parameters());
}
