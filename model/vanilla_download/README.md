| Feature            | fp16 (float16)                | bf16 (bfloat16)                    |
|--------------------|-------------------------------|------------------------------------|
| Bit size           | 16 bits                       | 16 bits                            |
| Precision          | Higher                        | Slightly lower                     |
| Range              | Smaller (~6.5e4)              | Larger (same as float32 ~3.4e38)   |
| Stability          | Needs loss scaling            | Stable, no loss scaling needed     |
| Hardware support   | Most GPUs (e.g., RTX, V100)   | A100, H100, TPUs, newer support    |
| Use in PyTorch     | fp16=True + loss_scale        | bf16=True                          |

If on A100 (like on SOL), bf16 is better: it's faster, more stable, and doesn't need loss scaling.