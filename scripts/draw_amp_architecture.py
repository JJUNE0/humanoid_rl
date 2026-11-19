"""Render architecture figures from the actual checkpoint configuration (Graphviz)."""

import argparse
import json
import subprocess
from pathlib import Path

from humanoid_rl.algorithm.amp.features import feature_schema

COLORS = {
    "actor": "#e8f2ff",
    "critic": "#eee8fa",
    "amp": "#fff0dc",
    "env": "#e7f5ed",
    "data": "#f1f4f8",
    "reward": "#fff8cc",
}


def amp_schema(meta):
    _, groups = feature_schema(meta["amp"]["config"].get("observe_object", False))
    groups = meta["amp"].get("feature_groups", groups)
    return sum(groups.values()), groups


class Diagram:
    def __init__(self, title, direction="LR"):
        self.lines = [
            "digraph G {",
            f"rankdir={direction};",
            'graph [bgcolor="white", pad="0.4", nodesep="0.45", ranksep="0.65", '
            'fontname="Noto Sans CJK KR", fontsize=24, labelloc=t];',
            'node [shape=box, style="rounded,filled", color="#61718b", '
            'fontname="Noto Sans CJK KR", fontsize=15, fontcolor="#16243b", margin="0.2,0.15"];',
            'edge [color="#526985", fontname="Noto Sans CJK KR", fontsize=12, arrowsize=0.8];',
            f"label={json.dumps(title, ensure_ascii=False)};",
        ]

    def node(self, name, label, color="data"):
        self.lines.append(
            f'{name} [label={json.dumps(label, ensure_ascii=False)}, fillcolor="{COLORS[color]}"];'
        )

    def edge(self, source, target, label="", **attributes):
        attributes["label"] = label
        attrs = ",".join(
            f"{k}={json.dumps(str(v), ensure_ascii=False)}"
            for k, v in attributes.items()
        )
        self.lines.append(f"{source} -> {target} [{attrs}];")

    def cluster(self, name, label, color="data"):
        self.lines.append(
            f"subgraph cluster_{name} {{ label={json.dumps(label, ensure_ascii=False)}; "
            f'color="{COLORS[color]}"; style="rounded"; fontsize=20;'
        )

    def end_cluster(self):
        self.lines.append("}")

    def rank(self, *names):
        self.lines.append("{rank=same; " + "; ".join(names) + ";}")

    def write(self, directory, name):
        path = directory / f"{name}.dot"
        path.write_text("\n".join(self.lines + ["}"]) + "\n")
        for extension in ("svg", "png"):
            subprocess.run(
                [
                    "dot",
                    f"-T{extension}",
                    "-Gdpi=144",
                    str(path),
                    "-o",
                    str(directory / f"{name}.{extension}"),
                ],
                check=True,
            )


def overall(meta):
    obs, action = meta["observation_dim"], meta["action_dim"]
    control = meta["amp"]["config"].get("control")
    feature_dim, _ = amp_schema(meta)
    observe_object = meta["amp"]["config"].get("observe_object", False)
    diagram = Diagram(
        "G1 · mjlab · FlashSAC + AMP\n" + ("별도 control critic / task reward = 0 / FB 없음" if control else "AMP reward만 사용 / task reward = 0"),
        direction="TB",
    )
    diagram.node(
        "env",
        "mjlab G1 환경\n29관절 PD + 물리 시뮬레이션\n박스·받침대 랜덤화 / 50 Hz",
        "env",
    )
    diagram.node(
        "obs",
        f"Actor observation: {obs}\nproprioception + box pose\n+ goal position + previous action" + ("\n+ hand/foot relative XYZ 12" if meta.get("observe_hand_feet") else "") + ("\n+ frozen stage context 28" if meta.get("stage_model") else ""),
        "env",
    )
    diagram.node(
        "actor",
        f"Actor πθ\n입력 {obs} → action {action}\n학습: 탐색 / play: tanh(mean)",
        "actor",
    )
    diagram.node(
        "replay",
        f"동일 index의 replay 저장\n(o, a, o′, terminated, truncated)\n+ Φ(s), Φ(s′) 각각 {feature_dim}",
        "data",
    )
    diagram.node(
        "expert",
        f"Expert train: {len(meta['amp']['train_clips'])} clips\n50 Hz motion.npz\nphase 균형 sampling",
        "data",
    )
    diagram.node(
        "phi",
        f"공통 feature Φ: {feature_dim} / pair: {2 * feature_dim}\nheading 불변 + 동일 mjlab G1 FK\n" + ("로봇 + 상자 pose·상대속도·손–상자 벡터\ngoal·contact·action·stage 조건 제외" if observe_object else "로봇 동작만; box·goal·action 제외"),
        "amp",
    )
    diagram.node(
        "dloss",
        "Discriminator loss\nexpert target +1 / policy target −1\nleast squares + expert gradient penalty",
        "amp",
    )
    diagram.node(
        "disc",
        f"현재 Discriminator Dψ\n{2 * feature_dim} → " + " → ".join(map(str, meta["amp"]["config"]["hidden_dims"])) + " → 1\nFP32 / score는 확률이 아님",
        "amp",
    )
    diagram.node(
        "reward",
        "AMP reward 재계산\nmax(0, 1 − 0.25(D − 1)²)\nreplay의 과거 reward를 덮어씀",
        "reward",
    )
    diagram.node(
        "norm",
        "FlashSAC reward normalization\n현재 수집 reward의 return 통계\ncritic support: [−5, +5]",
        "reward",
    )
    diagram.node(
        "critic",
        f"Double categorical critic Q1, Q2\n입력 o({obs}) + a({action})\n각 head: 101 bins → scalar Q",
        "critic",
    )
    diagram.node(
        "target",
        "Target Q1, Q2\ncritic과 동일 구조\noptimizer 없음 / EMA τ=0.01",
        "critic",
    )
    diagram.node(
        "alpha",
        "Temperature α = exp(log α)\n학습되는 scalar / 초기 0.01\npolicy entropy로 업데이트",
        "actor",
    )
    diagram.node(
        "validation",
        f"Held-out validation: {len(meta['amp']['validation_clips'])} clips\nscore·reward·feature distance 측정\noptimizer 업데이트에 사용하지 않음",
        "data",
    )
    diagram.edge("env", "obs")
    diagram.edge("obs", "actor")
    if meta.get("stage_model"):
        diagram.node("stage_mlp", "Frozen carry MLP\n원본 규약 state101 → context28\nstage9 / skill9 / progress9 / current failure1\n정규화·확률 보정·가중치 고정", "data")
        diagram.edge("env", "stage_mlp", "상자 body-frame 변환 + env origin 제거")
        diagram.edge("stage_mlp", "obs", "관측에만 추가 / task reward 없음")
    diagram.edge("actor", "env", "action", constraint="false", color="#247254")
    diagram.edge("env", "replay", "terminal 상태를 reset 전에 저장")
    diagram.edge("expert", "phi", "동일 feature 변환")
    diagram.edge("env", "phi", "s, s′", constraint="false")
    diagram.edge("phi", "dloss", "expert pair")
    diagram.edge("replay", "dloss", "policy pair")
    diagram.edge("dloss", "disc", "ψ optimizer step")
    diagram.edge("replay", "disc", "SAC batch의 pair")
    diagram.edge("disc", "reward", "no_grad")
    diagram.edge("reward", "norm")
    diagram.edge("norm", "critic", "TD target reward")
    diagram.edge("actor", "critic", "sampled action / log π", constraint="false")
    diagram.edge(
        "critic",
        "actor",
        "actor loss: α log π − min Q",
        constraint="false",
        color="#7254a2",
    )
    diagram.edge("critic", "target", "EMA", constraint="false")
    diagram.edge("target", "critic", "next-state TD distribution", constraint="false")
    diagram.edge("actor", "alpha", "entropy", constraint="false")
    diagram.edge("alpha", "critic", "entropy term", constraint="false")
    if control:
        diagram.node("cost", "제어 비용 6개 (reset 전)\naction rate / feet orientation / ankle roll\njoint limit / foot slip / undesired contact", "env")
        diagram.node("auxnorm", "가중 음수 비용 → EMA std 정규화\nmean subtraction 없음 / decay=0.99", "reward")
        diagram.node("auxq", f"별도 scalar twin critic Q_aux\n입력 o({obs}) + a({action}) → Q 2개\nFlashSAC trunk + unbounded scalar head", "critic")
        diagram.node("auxtarget", "Target Q_aux: parameter EMA\ny = r_aux + γ(1−terminated) min Q_aux_next\nentropy bonus 없음", "critic")
        diagram.edge("env", "cost")
        diagram.edge("cost", "replay", "동일 ring index 저장")
        diagram.edge("replay", "auxnorm", "6개 raw cost")
        diagram.edge("auxnorm", "auxq", "별도 TD 학습")
        diagram.edge("auxq", "auxtarget", "EMA τ=0.01", constraint="false")
        diagram.edge("auxtarget", "auxq", "bootstrap", constraint="false")
        diagram.edge("auxq", "actor", "+ 0.02 · Q 크기 보정 · Q_aux\ncritic freeze; dQ/da 전달", constraint="false", color="#7254a2")
    diagram.edge("validation", "disc", "평가만", style="dashed", constraint="false")
    diagram.edge(
        "expert",
        "env",
        ("일관된 scene reference reset: 50%\n로봇·상자·받침대·목표·상자 크기/질량 복원" if observe_object else "robot-only reference reset: 50%\n박스 파지는 강제하지 않음"),
        style="dashed",
        constraint="false",
    )
    diagram.rank("env", "obs", "actor")
    diagram.rank("expert", "phi", "replay", "validation")
    diagram.rank("dloss", "disc")
    diagram.rank("reward", "norm", "critic")
    diagram.rank("alpha", "target")
    return diagram


def auxiliary(meta):
    cfg = meta["agent"]
    control = meta["amp"]["config"]["control"]
    obs, action = meta["observation_dim"], meta["action_dim"]
    diagram = Diagram("Control critic 입출력 · FB 없음\nAMP 보상과 별도의 제어 비용 학습", direction="TB")
    diagram.node("costs", "환경에서 계산: reset 전\nΔaction², 발 기울기, 발목 roll²\n95% joint limit 초과, 발 속도 × 접촉, 비희망 몸체 접촉", "env")
    diagram.node("weight", "6개 raw 비용\n가중치 [−0.1, −0.4, −4, −10, −4, −1]\n합산 r_aux ≤ 0", "env")
    diagram.node("norm", "EMA variance, decay 0.99\nr_aux / std(r_aux)\nmean subtraction 없음", "reward")
    diagram.node("input", f"concat observation + action\n{obs} + {action} = {obs + action}\nQ1_aux, Q2_aux 독립 ensemble", "critic")
    diagram.node("trunk", f"FlashSAC critic trunk 재사용\nembed → {cfg['critic_num_blocks']} blocks (width {cfg['critic_hidden_dim']})\nUnitBatchNorm / residual / UnitRMSNorm", "critic")
    diagram.node("head", "각 critic: scalar linear head\n출력 Q1_aux, Q2_aux\n음수 return: 범위 제한 없음", "critic")
    diagram.node("target", "Target twin critic, EMA τ=0.01\ny = normalized_cost + γ(1−terminated) min Q_target(o′,a′)\ntimeout bootstrap / success stop / entropy 제외", "critic")
    diagram.node("loss", "MSE(Q_aux, y)\n별도 Adam + FlashSAC LR schedule\ngrad norm clip 10", "data")
    diagram.node("actor", f"Actor 목적: max Q_AMP + c·min Q_aux + αH\nc = {control['actor_weight']} · stopgrad(mean|Q_AMP| / max(mean|Q_aux|, 1))\ncritic parameter freeze / action gradient만 전달", "actor")
    diagram.edge("costs", "weight")
    diagram.edge("weight", "norm")
    diagram.edge("input", "trunk")
    diagram.edge("trunk", "head")
    diagram.edge("head", "loss")
    diagram.edge("norm", "target")
    diagram.edge("target", "loss")
    diagram.edge("loss", "head", "optimizer", constraint="false")
    diagram.edge("head", "target", "parameter EMA", constraint="false")
    diagram.edge("head", "actor", "dQ/da")
    diagram.rank("costs", "input")
    diagram.rank("norm", "head")
    diagram.rank("target", "loss", "actor")
    return diagram


def actor_critic(meta):
    cfg = meta["agent"]
    obs, action = meta["observation_dim"], meta["action_dim"]
    actor_width, critic_width = cfg["actor_hidden_dim"], cfg["critic_hidden_dim"]
    diagram = Diagram(
        "FlashSAC 네트워크 입출력\n실제 checkpoint 구조 / 배치 차원 B는 생략"
    )
    diagram.cluster("actor", "Actor πθ", "actor")
    diagram.node(
        "ain",
        f"Observation {obs}\n관절 위치 offset 29 + 속도 29\nbase v 3 + ω 3 + gravity 3\nbox 상대 위치 3 + 회전 6D\ngoal 상대 위치 3 + 이전 action 29" + ("\n양손·양발 root 상대 XYZ 12" if meta.get("observe_hand_feet") else "") + ("\nFrozen stage9 + skill9 + progress9 + failure1" if meta.get("stage_model") else ""),
        "actor",
    )
    diagram.node(
        "aembed", f"UnitBatchNorm({obs})\nUnitLinear {obs} → {actor_width}", "actor"
    )
    diagram.node(
        "ablock",
        f"FlashSACBlock × {cfg['actor_num_blocks']}\n{actor_width} → {4 * actor_width} → {actor_width}\nBN + ReLU / residual add\n마지막 UnitRMSNorm({actor_width})",
        "actor",
    )
    diagram.node(
        "ahead",
        f"NormalTanhPolicy\nmean μ: {action}\nlog σ: {action}, bound [−10, 2]",
        "actor",
    )
    diagram.node(
        "aout",
        f"Action {action} ∈ [−1, 1]\n수집: tanh(μ + σ ε), 반복 탐색 noise\nplay: tanh(μ)\n학습 forward: action + log π",
        "actor",
    )
    diagram.edge("ain", "aembed")
    diagram.edge("aembed", "ablock")
    diagram.edge("ablock", "ahead")
    diagram.edge("ahead", "aout")
    diagram.end_cluster()
    diagram.cluster(
        "critic", "Double critic Q1, Q2 — 두 head를 ensemble 연산으로 구현", "critic"
    )
    diagram.node(
        "qin",
        f"concat(o, a): {obs + action}\nobservation {obs} + action {action}",
        "critic",
    )
    diagram.node(
        "qembed",
        f"EnsembleUnitBatchNorm\n각 head: UnitLinear {obs + action} → {critic_width}",
        "critic",
    )
    diagram.node(
        "qblock",
        f"EnsembleFlashSACBlock × {cfg['critic_num_blocks']}\n각 head: {critic_width} → {4 * critic_width} → {critic_width}\nBN + ReLU / residual add\n마지막 EnsembleUnitRMSNorm",
        "critic",
    )
    diagram.node(
        "qout",
        f"각 head: {cfg['critic_num_bins']} categorical bins\nsoftmax → p(z | o, a)\nQ1, Q2 = Σ p(z) z\nz ∈ [{cfg['critic_min_v']:g}, {cfg['critic_max_v']:g}] (정규화된 return)",
        "critic",
    )
    diagram.node(
        "qt",
        "Target Q1, Q2\n동일한 두-head 구조 / frozen gradient\nQ̄ ← (1−τ)Q̄ + τQ, τ=0.01",
        "critic",
    )
    diagram.edge("qin", "qembed")
    diagram.edge("qembed", "qblock")
    diagram.edge("qblock", "qout")
    diagram.edge("qout", "qt", "parameter EMA")
    diagram.end_cluster()
    diagram.cluster("temp", "Temperature — MLP가 아닌 학습 scalar", "reward")
    diagram.node(
        "tin", "policy entropy H\n+ target entropy\n(target σ = 0.15)", "reward"
    )
    diagram.node(
        "tlog", "parameter log α: 1\nα = exp(log α) > 0\n초기 α = 0.01", "reward"
    )
    diagram.node(
        "tloss", "Lα = α · (H − Htarget)\nactor·critic entropy 항에 α 전달", "reward"
    )
    diagram.edge("tin", "tloss")
    diagram.edge("tlog", "tloss")
    diagram.end_cluster()
    return diagram


def discriminator(meta):
    cfg = meta["amp"]["config"]
    hidden = cfg["hidden_dims"]
    feature_dim, groups = amp_schema(meta)
    observe_object = cfg.get("observe_object", False)
    diagram = Diagram(
        "AMP Discriminator · feature · reward\n" + ("로봇·상자 상호작용 prior / goal·contact·action·stage 조건은 입력하지 않음" if observe_object else "로봇 동작 prior / box·goal·contact·action·phase label은 입력하지 않음"),
        direction="TB",
    )
    fields = [f"단일 상태 Φ(s): {feature_dim}"]
    titles = [
        "root 높이",
        "heading 제거 root 회전 6D",
        "heading 기준 root 선속도",
        "heading 기준 root 각속도",
        "29관절 위치 (이름 순서 정합)",
        "29관절 속도",
        "양손·양발 상대 위치 (동일 G1 FK)",
        "heading 기준 상자 상대 위치",
        "heading 기준 상자 회전 6D",
        "heading 기준 상자−root 선속도",
        "heading 기준 상자−root 각속도",
        "양손목에서 상자 중심까지 벡터",
    ]
    for title, count in zip(titles, groups.values()):
        fields.append(f"{title}: {count}")
    diagram.node("features", "\n".join(fields), "amp")
    diagram.node(
        "pair",
        f"20 ms 인접 상태 쌍\n[Φ(s_t), Φ(s_t+1)]\n{feature_dim} + {feature_dim} = {2 * feature_dim}\nclip/reset 경계를 넘지 않음",
        "amp",
    )
    diagram.node(
        "normalize",
        f"Frozen expert-train normalizer\n각 {feature_dim}차원에 같은 mean/std\nstd ≥ {cfg['min_feature_std']:g} / clip [−10, 10]\nvalidation 통계를 사용하지 않음",
        "amp",
    )
    previous = 2 * feature_dim
    diagram.edge("features", "pair")
    diagram.edge("pair", "normalize")
    last = "normalize"
    for i, width in enumerate(hidden):
        name = f"hidden_{i}"
        diagram.node(name, f"Linear {previous} → {width}\nELU", "amp")
        diagram.edge(last, name)
        last, previous = name, width
    diagram.node(
        "score",
        f"Linear {previous} → 1\nraw score Dψ\nexpert target +1\npolicy target −1",
        "amp",
    )
    diagram.edge(last, "score")
    diagram.node(
        "loss",
        f"L_D = E[(D_expert−1)²]\n+ E[(D_policy+1)²]\n+ ({cfg['gradient_penalty']:g}/2) E[||∇x D_expert||²]\nAdam lr={cfg['learning_rate']:g} / FP32",
        "data",
    )
    diagram.node(
        "reward",
        "r_AMP = max(0, 1−0.25(D−1)²)\nrange [0, 1], scale=1\nD=+1 → 1 / D=−1 → 0",
        "reward",
    )
    diagram.node(
        "sac",
        "현재 D로 replay reward 재계산\nFlashSAC reward normalization\n1-step TD / task reward = 0\ncritic을 통해 actor 학습",
        "critic",
    )
    diagram.edge("score", "loss", "ψ만 업데이트")
    diagram.edge("loss", "score", "gradient", constraint="false")
    diagram.edge("score", "reward", "no_grad; score는 확률이 아님")
    diagram.edge("reward", "sac")
    diagram.rank("features", "pair", "normalize")
    diagram.rank(*[f"hidden_{i}" for i in range(len(hidden))], "score")
    diagram.rank("loss", "reward", "sac")
    return diagram


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=Path("docs/figures"))
    args = parser.parse_args()
    meta = json.loads((args.checkpoint / "run.json").read_text())
    if meta.get("reward_mode") != "amp":
        parser.error("Architecture figures require an AMP checkpoint")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, diagram in (
        ("amp_connections", overall(meta)),
        ("flashsac_network_io", actor_critic(meta)),
        ("amp_discriminator_io", discriminator(meta)),
    ):
        diagram.write(args.output_dir, name)
        print(args.output_dir / f"{name}.png")
    if meta["amp"]["config"].get("control"):
        auxiliary(meta).write(args.output_dir, "control_critic_io")
        print(args.output_dir / "control_critic_io.png")


if __name__ == "__main__":
    main()
