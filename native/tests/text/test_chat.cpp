// Chat render goldens: chat.ChatFormat.render ids/text/prefill or ChatError message and param; concurrent renders
// from several threads must agree with the serial ones (ChatFormat is shared by request threads).
#include "text/chat.h"
#include "text_test.h"

#include <atomic>
#include <thread>

using namespace ingine::text;
using Json = nlohmann::ordered_json;

int main() {
  const auto dir = test::model_dir();
  if (dir.empty()) return test::kSkip;
  const Json golden = test::load_golden("chat.json");
  const ChatFormat fmt = ChatFormat::load(dir);
  test::Checker c;
  for (const Json& req : golden) {
    const std::string note = req["note"].get<std::string>();
    ChatOptions options;
    options.reasoning_effort = req["reasoning_effort"];
    options.enable_thinking = req["enable_thinking"];
    options.template_kwargs = req["kwargs"];
    try {
      const RenderedPrompt p = fmt.render(req["messages"], req["tools"], options);
      if (!c.expect(req.contains("result"), note + ": rendered, expected error " + req.value("error", Json()).dump())) {
        continue;
      }
      const Json& r = req["result"];
      c.expect(p.text == r["text"].get<std::string>(),
               note + ": text\n  got  " + test::show(p.text) + "\n  want " + test::show(r["text"].get<std::string>()));
      c.expect(p.ids == r["ids"].get<std::vector<int32_t>>(), note + ": ids");
      c.expect(p.prefill_ids == r["prefill_ids"].get<std::vector<int32_t>>(), note + ": prefill_ids");
    } catch (const ChatError& e) {
      if (!c.expect(req.contains("error"), note + ": ChatError " + test::show(e.what()) + " param " + e.param)) continue;
      const Json& err = req["error"];
      c.expect(e.what() == err["message"].get<std::string>(),
               note + ": message\n  got  " + test::show(e.what()) + "\n  want " +
                   test::show(err["message"].get<std::string>()));
      c.expect(e.param == err["param"].get<std::string>(), note + ": param " + e.param);
    }
  }
  // Concurrency: every thread renders every request and compares text and ids with a serial render (the
  // render cache is shared by the threads).
  const auto fingerprint = [](const RenderedPrompt& p) {
    std::string s = p.text + "|";
    for (const int32_t id : p.ids) s += std::to_string(id) + ",";
    return s;
  };
  std::vector<std::string> serial;
  for (const Json& req : golden) {
    ChatOptions options{req["reasoning_effort"], req["enable_thinking"], req["kwargs"]};
    try {
      serial.push_back(fingerprint(fmt.render(req["messages"], req["tools"], options)));
    } catch (const ChatError& e) {
      serial.push_back(std::string("error:") + e.what());
    }
  }
  std::atomic<int> mismatches{0};
  std::vector<std::thread> threads;
  for (std::size_t t = 0; t < 8; ++t) {
    threads.emplace_back([&, t] {
      for (int round = 0; round < 3; ++round) {
        for (std::size_t k = 0; k < golden.size(); ++k) {
          const std::size_t i = (k + t * 7) % golden.size();
          const Json& req = golden[i];
          ChatOptions options{req["reasoning_effort"], req["enable_thinking"], req["kwargs"]};
          std::string got;
          try {
            got = fingerprint(fmt.render(req["messages"], req["tools"], options));
          } catch (const ChatError& e) {
            got = std::string("error:") + e.what();
          }
          if (got != serial[i]) ++mismatches;
        }
      }
    });
  }
  for (auto& th : threads) th.join();
  c.expect(mismatches.load() == 0, "concurrent renders differ: " + std::to_string(mismatches.load()));
  return c.finish("text_chat");
}
