use std::path::Path;
use diffuser_error::DiffuserError;
use diffusers_load::{DiffuseTask};

fn main() -> Result<(), DiffuserError> {
    tracing_subscriber::registry();
    let mut args = DiffuseTask::default();
    args.positive_prompt = "A very realistic photo of a rusty robot walking on a sandy beach".to_string();
    args.negative_prompt = "robot".to_string();
    args.run(&Path::new("test.png"))
}
