# Server tests (CPU only). serve_scheduler: the scheduler with a multi-token-per-round generator.
# serve_server: the HTTP API end to end against FakeGenerator (needs the tokenizer and chat template of
# INGINE_MODEL_DIR, skipped otherwise).
ingine_add_test(NAME serve_scheduler SOURCES test_scheduler.cpp LIBS ingine_engine LABELS serve TIMEOUT 60)
ingine_add_test(NAME serve_server SOURCES test_server.cpp LIBS ingine_serve LABELS serve TIMEOUT 300)
