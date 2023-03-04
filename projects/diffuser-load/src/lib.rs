mod errors;

use safetensors::tensor::SafeTensors;
pub use errors::{Error, Result};



#[test]
fn test() {
    let tnesor = SafeTensors::deserialize(include_bytes!("../schoolAnime_schoolAnime.safetensors")).unwrap();
    println!("{:?}", tnesor.names());
}