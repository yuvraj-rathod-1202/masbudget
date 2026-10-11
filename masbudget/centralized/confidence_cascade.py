from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("confidence_cascade")
class ConfidenceCascadePolicy(BaseCentralizedPolicy):
    """Confidence-based cascade (FrugalGPT).

    Send each task (job) to the cheapest model first and escalate to the next more expensive
    model only if verification fails (quality below ``params["threshold"]``), until a model
    passes or every model has been tried. Every attempt is paid for, and the job keeps the
    answer of the last model it tried.

    If the budget cannot cover every cascade, each job first gets its cheapest attempt (while
    money lasts), then the failing job whose next attempt adds the most quality per unit cost
    is escalated, repeatedly.
    """

    def allocate(self, agents, models, budget, rng):
        jobs = agents.jobs
        if not jobs or budget <= 1e-9:
            return {j.job_id: FALLBACK for j in jobs}

        threshold = float(self.params.get("threshold", 0.8))
        ladder = sorted(models, key=lambda m: m.cost)
        if not ladder:
            return {j.job_id: FALLBACK for j in jobs}

        target_depth = {}
        for j in jobs:
            target_depth[j.job_id] = next(
                (k + 1 for k, m in enumerate(ladder) if m.quality(j.task) >= threshold), len(ladder)
            )

        full_cost = sum(m.cost for j in jobs for m in ladder[: target_depth[j.job_id]])
        if full_cost <= budget + 1e-9:
            return {j.job_id: [m.name for m in ladder[: target_depth[j.job_id]]] for j in jobs}

        # Under tight budget: give every job its cheapest attempt, then escalate by ROI
        depth = {j.job_id: 0 for j in jobs}
        rem = budget
        for j in jobs:
            if rem >= ladder[0].cost - 1e-9:
                depth[j.job_id] = 1
                rem -= ladder[0].cost

        while True:
            best_job = None
            best_ratio = -1.0
            for j in jobs:
                d = depth[j.job_id]
                if 0 < d < target_depth[j.job_id]:  # its last answer failed verification
                    nxt_m = ladder[d]
                    if rem >= nxt_m.cost - 1e-9:
                        delta_q = nxt_m.quality(j.task) - ladder[d - 1].quality(j.task)
                        ratio = delta_q / max(nxt_m.cost, 1e-9)  # the whole next call is paid
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_job = j.job_id
            if best_job is not None and best_ratio >= 0:
                rem -= ladder[depth[best_job]].cost
                depth[best_job] += 1
            else:
                break

        return {jid: [m.name for m in ladder[:d]] if d else FALLBACK for jid, d in depth.items()}
