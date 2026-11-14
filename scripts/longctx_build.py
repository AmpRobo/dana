"""Build long-context (up to 4096-token) train/val/test sets from teacher-labeled queries.

Each example keeps the target tool's own pool (the context the teacher labeled in) and adds
distractor tools from other pools until the tools JSON reaches a sampled token length.
Distractors that could also answer the query (shared name words or a known confusable group)
are excluded, so the short-context label stays correct in the long context.
"""
import glob, json, os, random

from dana import get_tokenizer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = f"{ROOT}/data/longctx"
MAX_TOOLS_TOKENS = 3950  # leaves room for query + <tools> under 4096
GENERIC = set("set get check toggle start stop create delete open search play find add close take send save run "
              "make share log enable request show list track control edit switch to my all on the of".split())
GROUPS = [
    "send_instant_message send_sms send_direct_message send_email compose_email send_family_broadcast",
    "convert_currency get_exchange_rate evaluate_js unit_convert cooking_convert",
    "pause_all_notifications toggle_do_not_disturb start_focus_mode clear_notifications",
    "check_screen_time get_daily_summary set_child_screen_time check_app_usage",
    "set_step_goal set_activity_goal get_step_count check_activity_progress",
    "get_fuel_level get_range check_car_status check_battery",
    "send_safety_check request_check_in", "share_photo share_file share_post",
    "open_file open_ebook open_app open_gallery",
    "get_directions start_navigation get_eta get_traffic find_my_car",
    "find_nearby find_ev_charger", "screen_mirror cast_media switch_tv_input",
    "start_screen_recording record_video stop_screen_recording",
    "car_climate set_thermostat", "get_indoor_temperature get_weather get_weather_forecast",
    "get_news_headlines get_trending_topics", "read_aloud listen_to_article",
    "toggle_night_mode toggle_dark_mode toggle_bedtime_mode",
    "play_show play_music search_streaming play_playlist",
    "web_search search_product search_recipe search_streaming search_photos search_email search_notes "
    "search_flights search_hotels search_for_contact define_word",
    "create_reminder create_list_item create_calendar_event create_note",
    "make_phone_call start_video_call call_emergency_services trigger_emergency_sos",
    "set_wallpaper change_watch_face", "kill_process close_app",
    "check_order_status track_food_delivery", "order_food add_to_cart",
    "request_ride book_rental_car", "get_upcoming_events get_trip_itinerary",
    "save_article bookmark_page save_location", "tv_power restart_device",
    "set_volume set_tv_volume", "check_heart_rate get_heart_rate_history",
]


def words(name):
    return set(name.split("_")) - GENERIC


def main():
    rng = random.Random(0)
    tok = get_tokenizer()
    pools = {os.path.basename(f)[:-5]: json.load(open(f)) for f in sorted(glob.glob(f"{ROOT}/examples/tools/*.json"))}
    all_tools = [(p, t) for p, ts in pools.items() for t in ts]
    cost = {t["name"]: len(tok.encode(json.dumps(t, separators=(",", ":")))) + 1 for _, t in all_tools}
    group_of = {}
    for g in GROUPS:
        for n in g.split():
            group_of.setdefault(n, set()).update(g.split())

    def conflicts(target, name):
        return bool(words(target) & words(name)) or name in group_of.get(target, ())

    def context(row, length, tail=False):
        own = list(pools[row["pool"]])
        used = sum(cost[t["name"]] for t in own)
        cands = [t for p, t in all_tools if p != row["pool"] and not conflicts(row["tool"], t["name"])]
        rng.shuffle(cands)
        extra = []
        for t in cands:
            if used + cost[t["name"]] > length:
                break
            extra.append(t)
            used += cost[t["name"]]
        if tail:  # target sits in the last 10% of the list, everything else shuffled before it
            rest = [t for t in own + extra if t["name"] != row["tool"]]
            rng.shuffle(rest)
            target = next(t for t in own if t["name"] == row["tool"])
            k = len(rest) - rng.randint(0, max(0, len(rest) // 10))
            tools = rest[:k] + [target] + rest[k:]
        else:
            tools = own + extra
            rng.shuffle(tools)
        return json.dumps(tools, separators=(",", ":")), used

    rows = [json.loads(l) for l in open(f"{OUT}/labeled.jsonl")]
    rows = [r for r in rows if r["ok"]]
    train_rows = [r for r in rows if r["split"] == "train"]
    test_rows = [r for r in rows if r["split"] == "test"]

    def ex(row, tools, n, bucket=None):
        e = {"query": row["query"], "tools": tools, "answers": row["answers"], "tool": row["tool"], "tokens": n}
        if bucket:
            e["bucket"] = bucket
        return e

    train = []
    for r in train_rows:
        for v in range(4):
            length = 0 if v == 0 else rng.randint(1024, MAX_TOOLS_TOKENS)  # 1 short + 3 long per query
            train.append(ex(r, *context(r, length)))
    rng.shuffle(train)

    buckets = {"short": 0, "2k": 2000, "4k": MAX_TOOLS_TOKENS}
    test = [ex(r, *context(r, n), bucket=b) for b, n in buckets.items() for r in test_rows]
    test += [ex(r, *context(r, MAX_TOOLS_TOKENS, tail=True), bucket="4k_tail") for r in test_rows]
    val = [ex(r, *context(r, rng.randint(1024, MAX_TOOLS_TOKENS))) for r in rng.sample(test_rows, 100)]

    for name, data in [("train", train), ("val", val), ("test", test)]:
        with open(f"{OUT}/{name}.jsonl", "w") as f:
            f.writelines(json.dumps(e) + "\n" for e in data)
        lens = sorted(e["tokens"] for e in data)
        print(f"{name}: {len(data)} examples, tools tokens median {lens[len(lens) // 2]}, max {lens[-1]}")


if __name__ == "__main__":
    main()
