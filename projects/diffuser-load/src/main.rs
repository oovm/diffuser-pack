use diffusers_load::{DiffuseTask, run};

fn main() -> anyhow::Result<()> {
    tracing_subscriber::registry();
    let mut args = DiffuseTask::default();
    args.prompt_positive = "A very realistic photo of a rusty robot walking on a sandy beach".to_string();
    run(args)
}
