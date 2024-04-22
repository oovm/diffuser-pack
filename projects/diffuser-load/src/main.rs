use std::path::Path;
use diffusers_load::{DiffuseTask, run};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry();
    let mut args = DiffuseTask::default();
    args.positive_prompt = "A very realistic photo of a rusty robot walking on a sandy beach".to_string();
    args.negative_prompt = "robot".to_string();
    run(args, &Path::new("test.png"))
}
