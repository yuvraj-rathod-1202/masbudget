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

    Optional params: ``verify_key`` - verification reads ``task.metadata[verify_key][model]``
    (e.g. realized unit-test outcomes) instead of ``q_{j,m}``, which then only ranks escalations;
    ``skip_dominated`` - drop models that cost more than another model but have no higher mean
    quality over the jobs; ``escalation: "jump"`` - instead of always trying the next model of
    the ladder, a failing job may escalate to any stronger model: every job gets its cheapest
    attempt, then the (failing job, stronger model) pair with the highest q gain per cost of
    the call is tried, repeatedly, while some escalation with a positive gain is affordable.
    """

    def allocate(self, agents, models, budget, rng):
        jobs = agents.jobs
        if not jobs or budget <= 1e-9:
            return {j.job_id: FALLBACK for j in jobs}

        threshold = float(self.params.get("threshold", 0.8))
        verify_key = self.params.get("verify_key")
        ladder = sorted(models, key=lambda m: m.cost)
        if self.params.get("skip_dominated", False):
            mean_q = {m.name: sum(m.quality(j.task) for j in jobs) / len(jobs) for m in ladder}
            ladder = [m for k, m in enumerate(ladder) if all(mean_q[m.name] > mean_q[p.name] for p in ladder[:k])]
        if not ladder:
            return {j.job_id: FALLBACK for j in jobs}

        def passes(job, model):
            score = job.task.metadata[verify_key][model.name] if verify_key else model.quality(job.task)
            return score >= threshold

        if self.params.get("escalation", "ladder") == "jump":
            return self._jump(jobs, ladder, budget, passes)

        target_depth = {}
        for j in jobs:
            target_depth[j.job_id] = next(
                (k + 1 for k, m in enumerate(ladder) if passes(j, m)), len(ladder)
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

    @staticmethod
    def _jump(jobs, ladder, budget, passes):
        calls = {j.job_id: [] for j in jobs}
        rem = budget
        for j in jobs:
            if rem >= ladder[0].cost - 1e-9:
                calls[j.job_id].append(ladder[0])
                rem -= ladder[0].cost

        while True:
            best, best_ratio = None, 0.0
            for j in jobs:
                tried = calls[j.job_id]
                if not tried or passes(j, tried[-1]):
                    continue
                last = tried[-1]
                for m in ladder[ladder.index(last) + 1:]:
                    if m.cost <= rem + 1e-9:
                        ratio = (m.quality(j.task) - last.quality(j.task)) / max(m.cost, 1e-9)
                        if ratio > best_ratio:
                            best, best_ratio = (j.job_id, m), ratio
            if best is None:
                break
            calls[best[0]].append(best[1])
            rem -= best[1].cost

        return {jid: [m.name for m in tried] if tried else FALLBACK for jid, tried in calls.items()}
