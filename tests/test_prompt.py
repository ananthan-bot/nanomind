"""tests/test_prompt.py — Tests for prompt templates and chat format."""
import pytest
from nanomind.prompt import (
    Role, Message, Conversation,
    ChatMLTemplate, LLaMA2Template, LLaMA3Template,
    AlpacaTemplate, CompletionTemplate,
    FewShotBuilder, FewShotExample,
    PromptConfig, PromptManager,
    get_template, list_templates, register_template,
    PromptTemplate,
)

# ── Types ─────────────────────────────────────────────────────────────────────

class TestTypes:
    def test_message_to_dict(self):
        m = Message.user("hello")
        d = m.to_dict()
        assert d["role"] == "user"
        assert d["content"] == "hello"

    def test_conversation_add(self):
        c = Conversation()
        c.user("hi").assistant("hello")
        assert len(c) == 2

    def test_conversation_to_list(self):
        c = Conversation()
        c.user("q")
        lst = c.to_list()
        assert isinstance(lst, list)
        assert lst[0]["role"] == "user"

    def test_last_user_message(self):
        c = Conversation()
        c.user("first").assistant("a").user("second")
        assert c.last_user_message().content == "second"


# ── ChatML ────────────────────────────────────────────────────────────────────

class TestChatML:
    def test_contains_im_start(self):
        t    = ChatMLTemplate()
        conv = Conversation()
        conv.user("hi")
        out  = t.render(conv)
        assert "<|im_start|>" in out

    def test_contains_im_end(self):
        t    = ChatMLTemplate()
        conv = Conversation()
        conv.user("hi")
        out  = t.render(conv)
        assert "<|im_end|>" in out

    def test_generation_prompt(self):
        t    = ChatMLTemplate(add_generation_prompt=True)
        conv = Conversation(); conv.user("hi")
        assert t.render(conv).endswith("assistant
")

    def test_no_generation_prompt(self):
        t    = ChatMLTemplate(add_generation_prompt=False)
        conv = Conversation(); conv.user("hi")
        assert not t.render(conv).endswith("assistant
")

    def test_system_included(self):
        t    = ChatMLTemplate()
        conv = Conversation(system_prompt="Be helpful.")
        conv.user("hi")
        out  = t.render(conv)
        assert "Be helpful." in out

    def test_apply_convenience(self):
        t   = ChatMLTemplate()
        out = t.apply(system="sys", user="usr")
        assert "sys" in out and "usr" in out


# ── LLaMA ─────────────────────────────────────────────────────────────────────

class TestLLaMA:
    def test_llama2_inst_tags(self):
        t    = LLaMA2Template()
        conv = Conversation(); conv.user("hello")
        out  = t.render(conv)
        assert "[INST]" in out and "[/INST]" in out

    def test_llama2_system_tag(self):
        t    = LLaMA2Template()
        conv = Conversation(system_prompt="You are NanoMind.")
        conv.user("hi")
        out  = t.render(conv)
        assert "<<SYS>>" in out

    def test_llama3_begin_token(self):
        t    = LLaMA3Template()
        conv = Conversation(); conv.user("hi")
        out  = t.render(conv)
        assert out.startswith("<|begin_of_text|>")

    def test_llama3_header_tokens(self):
        t    = LLaMA3Template()
        conv = Conversation(); conv.user("hi")
        out  = t.render(conv)
        assert "<|start_header_id|>" in out


# ── Alpaca + Completion ───────────────────────────────────────────────────────

class TestAlpacaCompletion:
    def test_alpaca_instruction_header(self):
        t    = AlpacaTemplate()
        conv = Conversation(); conv.user("Do X")
        out  = t.render(conv)
        assert "### Instruction:" in out

    def test_alpaca_response_header(self):
        t    = AlpacaTemplate()
        conv = Conversation(); conv.user("Do X")
        out  = t.render(conv)
        assert "### Response:" in out

    def test_alpaca_format_instruction(self):
        t   = AlpacaTemplate()
        out = t.format_instruction("Sort this list", "3,1,2")
        assert "### Input:" in out

    def test_completion_human_prefix(self):
        t    = CompletionTemplate()
        conv = Conversation(); conv.user("hi")
        out  = t.render(conv)
        assert "Human:" in out

    def test_completion_assistant_prefix(self):
        t    = CompletionTemplate()
        conv = Conversation(); conv.user("hi"); conv.assistant("hello")
        out  = t.render(conv)
        assert "Assistant:" in out


# ── FewShotBuilder ────────────────────────────────────────────────────────────

class TestFewShot:
    def _builder(self):
        return FewShotBuilder(
            CompletionTemplate(),
            examples=[FewShotExample("2+2", "4"), FewShotExample("3+3", "6")],
            input_key="Q", output_key="A",
        )

    def test_len(self):
        b = self._builder()
        assert len(b) == 2

    def test_build_contains_examples(self):
        b   = self._builder()
        out = b.build("5+5")
        assert "2+2" in out and "3+3" in out

    def test_n_shots_limit(self):
        b   = self._builder()
        out = b.build("5+5", n_shots=1)
        assert "2+2" in out
        assert "3+3" not in out

    def test_add_example(self):
        b = self._builder()
        b.add_example("4+4", "8")
        assert len(b) == 3

    def test_query_in_output(self):
        b   = self._builder()
        out = b.build("10+10")
        assert "10+10" in out


# ── Registry ──────────────────────────────────────────────────────────────────

class TestRegistry:
    def test_list_templates(self):
        templates = list_templates()
        assert "chatml" in templates
        assert "llama2" in templates

    def test_get_template(self):
        t = get_template("chatml")
        assert isinstance(t, ChatMLTemplate)

    def test_unknown_template_raises(self):
        with pytest.raises(KeyError):
            get_template("unknown_format_xyz")

    def test_register_custom(self):
        class MyTemplate(PromptTemplate):
            name = "custom_test"
            def render(self, c): return "custom"
            def render_message(self, m): return m.content
        register_template("custom_test", MyTemplate)
        t = get_template("custom_test")
        assert isinstance(t, MyTemplate)


# ── PromptConfig ──────────────────────────────────────────────────────────────

class TestPromptConfig:
    def test_defaults(self):
        cfg = PromptConfig()
        assert cfg.template == "chatml"

    def test_invalid_template(self):
        with pytest.raises(AssertionError):
            PromptConfig(template="gpt3")

    def test_invalid_history_turns(self):
        with pytest.raises(AssertionError):
            PromptConfig(max_history_turns=0)


# ── PromptManager ─────────────────────────────────────────────────────────────

class TestPromptManager:
    def test_build_returns_string(self):
        pm  = PromptManager()
        out = pm.build("hello")
        assert isinstance(out, str)
        assert "hello" in out

    def test_n_turns_increments(self):
        pm = PromptManager()
        pm.build("q1")
        pm.add_assistant("a1")
        pm.build("q2")
        assert pm.n_turns == 2

    def test_reset_clears_history(self):
        pm = PromptManager()
        pm.build("q1"); pm.add_assistant("a1")
        pm.reset()
        assert pm.n_turns == 0

    def test_max_history_truncates(self):
        pm = PromptManager(PromptConfig(max_history_turns=1))
        pm.build("q1"); pm.add_assistant("a1")
        pm.build("q2"); pm.add_assistant("a2")
        out = pm.build("q3")
        # Only last 1 turn (q2/a2) + current q3 should appear
        assert "q1" not in out

    def test_system_prompt_in_output(self):
        pm  = PromptManager(PromptConfig(system_prompt="You are NanoMind."))
        out = pm.build("hi")
        assert "NanoMind" in out

    def test_repr(self):
        pm  = PromptManager()
        assert "PromptManager" in repr(pm)


# ── render_messages helper ────────────────────────────────────────────────────

class TestRenderMessages:
    def test_render_empty(self):
        t   = ChatMLTemplate(add_generation_prompt=False)
        out = t.render_messages([])
        assert out == ""

    def test_render_multiple(self):
        t    = ChatMLTemplate(add_generation_prompt=False)
        msgs = [Message.user("hi"), Message.assistant("hello")]
        out  = t.render_messages(msgs)
        assert "hi" in out and "hello" in out

    def test_completion_separator(self):
        t    = CompletionTemplate(sep="\n---\n")
        conv = Conversation(); conv.user("q1"); conv.assistant("a1"); conv.user("q2")
        out  = t.render(conv)
        assert "---" in out
