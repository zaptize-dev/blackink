// StreamParser goldens: per-token Delta sequences, finished flags, finish(), Completion and merge_deltas.
#include "text/chat.h"
#include "text_test.h"

using namespace ingine::text;
using Json = nlohmann::ordered_json;

namespace {

Json to_json(const Delta& d) {
  Json j = Json::object();
  j["reasoning_content"] = d.reasoning_content;
  j["content"] = d.content;
  if (d.tool_call) {
    Json tc = Json::object();
    tc["index"] = d.tool_call->index;
    tc["id"] = d.tool_call->id ? Json(*d.tool_call->id) : Json(nullptr);
    tc["name"] = d.tool_call->name ? Json(*d.tool_call->name) : Json(nullptr);
    tc["arguments"] = d.tool_call->arguments;
    j["tool_call"] = std::move(tc);
  } else {
    j["tool_call"] = nullptr;
  }
  return j;
}

Json to_json(const std::vector<Delta>& ds) {
  Json j = Json::array();
  for (const Delta& d : ds) j.push_back(to_json(d));
  return j;
}

}  // namespace

int main() {
  const auto dir = test::model_dir();
  if (dir.empty()) return test::kSkip;
  const Json golden = test::load_golden("stream.json");
  const ChatFormat fmt = ChatFormat::load(dir);
  test::Checker c;
  for (const Json& cs : golden) {
    const std::string note = cs["note"].get<std::string>();
    std::vector<std::string> stop;
    if (cs["stop"].is_array()) stop = cs["stop"].get<std::vector<std::string>>();
    int counter = 0;
    const auto prefill = cs["prefill"].get<std::vector<int32_t>>();
    StreamParser p = fmt.parser(prefill, stop, [&counter] { return "call_" + std::to_string(counter++); });
    std::vector<Delta> all;
    const auto tokens = cs["tokens"].get<std::vector<int32_t>>();
    for (std::size_t i = 0; i < tokens.size(); ++i) {
      const auto ds = p.feed(tokens[i]);
      all.insert(all.end(), ds.begin(), ds.end());
      const Json& step = cs["steps"][i];
      const Json got = to_json(ds);
      c.expect(got == step["deltas"], note + ": step " + std::to_string(i) + "\n  got  " + got.dump() + "\n  want " +
                                          step["deltas"].dump());
      c.expect(p.finished() == step["finished"].get<bool>(), note + ": finished at step " + std::to_string(i));
    }
    const auto fin = p.finish();
    all.insert(all.end(), fin.begin(), fin.end());
    c.expect(to_json(fin) == cs["finish"], note + ": finish\n  got  " + to_json(fin).dump() + "\n  want " +
                                               cs["finish"].dump());
    c.expect(to_json(p.finish()) == cs["finish_again"], note + ": finish again");
    c.expect(p.num_tokens() == cs["num_tokens"].get<int>(), note + ": num_tokens");
    c.expect(p.finished() == cs["finished"].get<bool>(), note + ": finished");
    const Completion r = p.result();
    const Json& want = cs["result"];
    c.expect(r.reasoning_content == want["reasoning_content"].get<std::string>(), note + ": reasoning_content");
    c.expect(r.content == want["content"].get<std::string>(),
             note + ": content got " + test::show(r.content) + " want " + want["content"].dump());
    Json calls = Json::array();
    for (const ToolCall& tc : r.tool_calls) calls.push_back(Json{{"id", tc.id}, {"name", tc.name}, {"arguments", tc.arguments}});
    c.expect(calls == want["tool_calls"], note + ": tool_calls\n  got  " + calls.dump() + "\n  want " +
                                              want["tool_calls"].dump());
    c.expect(r.finish_reason == (want["finish_reason"].is_null() ? std::nullopt
                                                                  : std::optional(want["finish_reason"].get<std::string>())),
             note + ": finish_reason " + r.finish_reason.value_or("null"));
    c.expect(p.finish_reason() == r.finish_reason, note + ": finish_reason accessor");
    c.expect(to_json(merge_deltas(all)) == cs["merged"], note + ": merge_deltas");

    // Non-streaming parse stops at the first finished state and must agree with the stream.
    counter = 0;
    const Completion np = fmt.parse(tokens, prefill, stop, [&counter] { return "call_" + std::to_string(counter++); });
    c.expect(np.content == r.content && np.reasoning_content == r.reasoning_content && np.tool_calls == r.tool_calls &&
                 np.finish_reason == r.finish_reason,
             note + ": parse() agrees with the stream");
  }
  return c.finish("text_stream");
}
