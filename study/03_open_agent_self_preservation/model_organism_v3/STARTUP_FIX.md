# Adapter-target guard correction before optimization

The first correction fit (`r3-p2-fit-preservation`) stopped after 8.54 seconds, before any optimizer step or adapter export. Its entry-limit and provenance receipts are retained.

The guard incorrectly expected `peft_config.target_modules` to contain 186 fully qualified module paths. The saved PEFT configuration uses 12 module-type suffixes, which expand to the 186 actual language modules. This representation had already been used successfully for version-2 inference.

The corrected guard inspects the loaded modules carrying `lora_A` and `lora_B`, requires exactly 186 language modules, and requires exactly 5,411,328 trainable parameters. It still refuses any trainable non-LoRA parameter. Restarted fit jobs use a `-targets2` suffix. Source versions and the failed startup are preserved.

No curriculum, seed, optimizer setting, checkpoint-selection rule, or success threshold changed. The isolation boundary files also remain unchanged.
