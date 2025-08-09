1. idea - we have a custom embedding engine, which we can directly plug into llamaindex. We should convert this into a pluggable system so as to support using approaches like pylate/colbert.
2. file watching I believe does not detect new and may or may not detect delted files.
3. the rewrite agent needs to be tuned.
4. logging needs to be harmonized. some debug messages are info, far more the other way around. 