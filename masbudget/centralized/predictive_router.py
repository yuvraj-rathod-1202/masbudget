from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("predictive_router")
class PredictiveRouterPolicy(BaseCentralizedPolicy):
    """Predictive LLM router (RouteLLM / MasRouter style).

    Routes each task using a trained quality estimator over task features / difficulty,
    trading predicted quality gain against marginal cost, without solving an exact ILP.
    """

    def allocate(self, agents, models, budget, rng):
        if not agents or budget <= 1e-9:
            return {a.agent_id: FALLBACK for a in agents}

        cheapest = models.cheapest()
        mapping = {a.agent_id: FALLBACK for a in agents}
        rem = budget

        # Initial assignment: prioritize harder tasks when budget cannot cover all agents
        sorted_by_diff = sorted(agents, key=lambda a: a.task.difficulty_rank, reverse=True)
        for a in sorted_by_diff:
            if rem >= cheapest.cost - 1e-9:
                mapping[a.agent_id] = cheapest.name
                rem -= cheapest.cost

        # Greedy incremental upgrade: route tasks to stronger models based on marginal quality gain per cost
        while True:
            best_upgrade = None
            best_score = -1.0
            for a in agents:
                curr_model = models.get(mapping[a.agent_id])
                if curr_model.name == FALLBACK:
                    continue
                for m in models:
                    delta_c = m.cost - curr_model.cost
                    if delta_c > 1e-9 and rem >= delta_c - 1e-9:
                        delta_q = m.quality(a.task) - curr_model.quality(a.task)
                        if delta_q > 0:
                            score = (delta_q / delta_c) * (1.0 + 0.1 * a.task.difficulty_rank)
                            if score > best_score:
                                best_score = score
                                best_upgrade = (a.agent_id, m.name, delta_c)
            if best_upgrade is not None:
                aid, mname, delta_c = best_upgrade
                mapping[aid] = mname
                rem -= delta_c
            else:
                break

        return mapping

