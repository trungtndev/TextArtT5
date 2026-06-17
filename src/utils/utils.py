import torch
from torchmetrics import Metric
from pytorch_lightning.utilities import rank_zero_only

class MaskedAccuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("correct_masked_preds", default=torch.tensor(0), dist_reduce_fx="sum")
        self.add_state("total_masked_tokens", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds: torch.Tensor, target: torch.Tensor, mask: torch.Tensor):
        if preds.dim() == 3 and target.dim() == 2 and mask.dim() == 2:
            preds = preds.view(-1, preds.shape[-1])  # Reshape to (B*L, V)
            target = target.view(-1)  # Reshape to (B*L)
            mask = mask.view(-1)  # Reshape to (B*L)
        elif not (preds.dim() == 2 and target.dim() == 1 and mask.dim() == 1):
            raise ValueError("Input tensors preds, target, and mask must have compatible shapes.")

        preds_indices = torch.argmax(preds, dim=1)

        masked_preds = preds_indices[mask]
        masked_target = target[mask]

        self.correct_masked_preds += torch.sum(masked_preds == masked_target)
        self.total_masked_tokens += mask.sum()

    def compute(self) -> torch.Tensor:
        accuracy = self.correct_masked_preds.float() / self.total_masked_tokens
        return accuracy if self.total_masked_tokens > 0 else torch.tensor(0.0)


@rank_zero_only
def log_parameters(model: torch.nn.Module):
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params
    model_name = model.__class__.__name__

    RESET = "\033[0m"
    BOLD = "\033[1m"

    BORDER = "\033[96m"
    TEXT_MAIN = "\033[97m"

    TRAIN_C = "\033[92m"
    FROZEN_C = "\033[94m"
    PARTIAL_C = "\033[93m"

    TITLE_C = "\033[95m"

    w_name = 20
    w_status = 12
    w_param = 15

    header_inner = f" {'Module / Layer':<{w_name}} | {'Status':<{w_status}} | {'Trainable':>{w_param}} | {'Total':>{w_param}} "
    total_width = len(header_inner)

    def get_stats(m):
        t = sum(p.numel() for p in m.parameters())
        r = sum(p.numel() for p in m.parameters() if p.requires_grad)
        return t, r

    print("\n")

    print(f"{BORDER}╔{'═' * total_width}╗{RESET}")

    model_display = f"MODEL: {model_name}"
    pad_len = (total_width - len(model_display)) // 2
    extra_pad = 1 if (total_width - len(model_display)) % 2 != 0 else 0
    print(
        f"{BORDER}║{RESET}{' ' * pad_len}{BOLD}{TITLE_C}{model_display}{RESET}{' ' * (pad_len + extra_pad)}{BORDER}║{RESET}")

    print(f"{BORDER}╠{'═' * total_width}╣{RESET}")

    p_total = f"{BOLD}{TEXT_MAIN}{total_params:,}{RESET}"
    p_train = f"{BOLD}{TRAIN_C}{trainable_params:,}{RESET}"
    p_frozen = f"{BOLD}{FROZEN_C}{frozen_params:,}{RESET}"

    raw_stats_str = f"Total: {total_params:,} | Trainable: {trainable_params:,} | Frozen: {frozen_params:,}"
    pad_len = (total_width - len(raw_stats_str)) // 2
    extra_pad = 1 if (total_width - len(raw_stats_str)) % 2 != 0 else 0

    stats_colored = f"{TEXT_MAIN}Total: {p_total} {BORDER}|{RESET} {TEXT_MAIN}Trainable: {p_train} {BORDER}|{RESET} {TEXT_MAIN}Frozen: {p_frozen}"

    print(f"{BORDER}║{RESET}{' ' * pad_len}{stats_colored}{' ' * (pad_len + extra_pad)}{BORDER}║{RESET}")
    print(f"{BORDER}╠{'═' * total_width}╣{RESET}")

    print(
        f"{BORDER}║{RESET}{BOLD}{TEXT_MAIN} {'Module / Layer':<{w_name}} {BORDER}|{RESET}{TEXT_MAIN} {'Status':<{w_status}} {BORDER}|{RESET}{TEXT_MAIN} {'Trainable':>{w_param}} {BORDER}|{RESET}{TEXT_MAIN} {'Total':>{w_param}} {RESET}{BORDER}║{RESET}")
    print(f"{BORDER}╟{'─' * total_width}╢{RESET}")

    def inspect_recursive(module, name, level=0):
        t, r = get_stats(module)
        if t == 0: return

        if r == 0:
            status_txt = "FROZEN"
            row_color = FROZEN_C
            raw_status = "FROZEN"
        elif r == t:
            status_txt = "TRAIN"
            row_color = TRAIN_C
            raw_status = "TRAIN"
        else:
            status_txt = "PARTIAL"
            row_color = PARTIAL_C
            raw_status = "PARTIAL"

        indent = "  " * level + ("└─ " if level > 0 else "")
        display_name = f"{indent}{name}"
        if len(display_name) > w_name:
            display_name = display_name[:w_name - 3] + "..."

        print(
            f"{BORDER}║{RESET} "
            f"{TEXT_MAIN}{display_name:<{w_name}} {BORDER}|{RESET} "
            f"{row_color}{status_txt:<{w_status}}{RESET} {BORDER}|{RESET} "
            f"{row_color}{r:>{w_param},}{RESET} {BORDER}|{RESET} "
            f"{TEXT_MAIN}{t:>{w_param},}"
            f" {BORDER}║{RESET}"
        )

        if raw_status == "PARTIAL":
            for child_name, child in module.named_children():
                inspect_recursive(child, child_name, level + 1)

    for name, child in model.named_children():
        inspect_recursive(child, name, level=0)

    print(f"{BORDER}╚{'═' * total_width}╝{RESET}\n")
