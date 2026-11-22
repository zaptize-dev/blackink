# Text goldens (CPU): tokenizer, chat renders, StreamParser, Python-compat JSON helpers.
ingine_add_test(NAME text_tokenizer SOURCES test_tokenizer.cpp LIBS ingine_text TIMEOUT 300)
ingine_add_test(NAME text_chat SOURCES test_chat.cpp LIBS ingine_text TIMEOUT 300)
ingine_add_test(NAME text_stream SOURCES test_stream.cpp LIBS ingine_text TIMEOUT 300)
ingine_add_test(NAME text_pyjson SOURCES test_pyjson.cpp LIBS ingine_text)
# Render cache: incremental renders of growing, edited and interleaved conversations against cold renders.
ingine_add_test(NAME text_render_cache SOURCES test_render_cache.cpp LIBS ingine_text TIMEOUT 600)
# Captured chat requests (optional local data): cold vs incremental renders; skipped when no directory is set.
set(INGINE_TRACE_DIRS "" CACHE STRING "Directories of captured chat requests for text_traces (;-separated)")
ingine_add_test(NAME text_traces SOURCES trace_render.cpp LIBS ingine_text ARGS ${INGINE_TRACE_DIRS} TIMEOUT 600)
