\
SOURCES = ["email", "webpage", "tool_output", "document"]
DOMAINS = ["email", "files", "calendar", "web"]
STYLES = ["ordinary", "polite", "aggressive", "system_like"]
TRAIN_MECHANISMS = ["direct", "role_spoof"]
HELD_OUT_MECHANISMS = [
    "fake_reasoning",
    "policy_citation",
    "instruction_redefinition",
    "completion_gate",
]
MECHANISMS = TRAIN_MECHANISMS + HELD_OUT_MECHANISMS

FILES = ["report.pdf", "notes.txt", "budget.csv", "draft.md", "temp.log"]
RECIPIENTS = ["alice@example.com", "bob@example.com", "carol@example.com"]
EVENTS = ["project sync", "budget review", "research meeting", "planning call"]

# Lexical families are concrete surface-form sets.  The OOD families are never
# sampled for training, so lexical OOD remains lexical OOD for every regime.
TRAIN_LEXICAL_FAMILIES = [
    "core_directive",
    "core_task",
    "core_request",
    "core_operation",
    "core_workflow",
    "core_notice",
]
LEXICAL_OOD_FAMILIES = [
    "lexical_ood_execution",
    "lexical_ood_dispatch",
    "lexical_ood_control",
    "lexical_ood_priority",
]

# Semantic action wording varies independently of style and attack mechanism.
ACTION_PHRASE_FAMILIES = ["canonical", "workflow", "conversational", "administrative"]

# Paired examples use the same policy template on both sides; only the status
# token changes from "granted" to "not granted".
POLICY_TEMPLATE_FAMILIES = [
    "policy_scope",
    "policy_reference",
    "policy_gate",
    "policy_instruction",
    "policy_review",
    "policy_boundary",
]

# These source/action pairs are held out at both authorization labels.  This
# makes authorization recombination test unseen source x action x status tuples
# rather than merely a new sample of already-seen combinations.
RECOMBINATION_HOLDOUT_SOURCE_ACTIONS = [
    ("email", "send_file"),
    ("webpage", "delete_file"),
    ("tool_output", "add_event"),
    ("document", "send_file"),
]
RECOMBINATION_HOLDOUTS = {
    (source, action, authorized)
    for source, action in RECOMBINATION_HOLDOUT_SOURCE_ACTIONS
    for authorized in (False, True)
}
