from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("confidence_cascade")
class ConfidenceCascadePolicy(BaseCentralizedPolicy):
    """Confidence-based cascade (FrugalGPT).

    Send each task to the cheapest model first and escalate to a more capable model
    only if verification fails (e.g. quality below ``params["threshold"]``).
    """

    def allocate(self, agents, models, budget, rng):
        if not agents or budget <= 1e-9:
            return {a.agent_id: FALLBACK for a in agents}

        threshold = float(self.params.get("threshold", 0.8))
        sorted_models = sorted(models, key=lambda m: m.cost)
        if not sorted_models:
            return {a.agent_id: FALLBACK for a in agents}
        cheapest = sorted_models[0]

        # Determine target model for each agent: first model reaching threshold, or max quality
        targets = {}
        for a in agents:
            chosen = None
            for m in sorted_models:
                if m.quality(a.task) >= threshold:
                    chosen = m
                    break
            if chosen is None:
                chosen = max(sorted_models, key=lambda m: m.quality(a.task))
            targets[a.agent_id] = chosen

        target_cost = sum(m.cost for m in targets.values())
        if target_cost <= budget + 1e-9:
            return {a.agent_id: targets[a.agent_id].name for a in agents}

        # Under tight budget: allocate cheapest model, then escalate by ROI
        current = {a.agent_id: models.fallback for a in agents}
        rem = budget
        for a in agents:
            if rem >= cheapest.cost - 1e-9:
                current[a.agent_id] = cheapest
                rem -= cheapest.cost

        while True:
            best_upgrade = None
            best_ratio = -1.0
            for a in agents:
                cur_m = current[a.agent_id]
                tgt_m = targets[a.agent_id]
                if cur_m.name != tgt_m.name:
                    cur_idx = sorted_models.index(cur_m) if cur_m in sorted_models else -1
                    nxt_m = sorted_models[cur_idx + 1]
                    delta_c = nxt_m.cost - cur_m.cost
                    if rem >= delta_c - 1e-9:
                        delta_q = nxt_m.quality(a.task) - cur_m.quality(a.task)
                        ratio = delta_q / max(delta_c, 1e-9)
                        if ratio > best_ratio:
                            best_ratio = ratio
                            best_upgrade = (a.agent_id, nxt_m, delta_c)
            if best_upgrade is not None and best_ratio >= 0:
                aid, nxt_m, delta_c = best_upgrade
                current[aid] = nxt_m
                rem -= delta_c
            else:
                break

        return {aid: m.name for aid, m in current.items()}

