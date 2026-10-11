from masbudget.centralized.base import CENTRALIZED_POLICIES, BaseCentralizedPolicy


from masbudget.core import FALLBACK


@CENTRALIZED_POLICIES.register("predictive_router")
class PredictiveRouterPolicy(BaseCentralizedPolicy):
    """Predictive LLM router (RouteLLM / MasRouter style).

    Routes each task using a trained quality estimator over task features / difficulty,
    trading predicted quality gain against marginal cost, without solving an exact ILP.
    """

    def allocate(self, agents, models, budget, rng):
        jobs = agents.jobs
        if not jobs or budget <= 1e-9:
            return {j.job_id: FALLBACK for j in jobs}

        cheapest = models.cheapest()
        mapping = {j.job_id: FALLBACK for j in jobs}
        rem = budget

        # Initial assignment: prioritize harder tasks when budget cannot cover all jobs
        sorted_by_diff = sorted(jobs, key=lambda j: j.task.difficulty_rank, reverse=True)
        for j in sorted_by_diff:
            if rem >= cheapest.cost - 1e-9:
                mapping[j.job_id] = cheapest.name
                rem -= cheapest.cost

        # Greedy incremental upgrade: route tasks to stronger models based on marginal quality gain per cost
        while True:
            best_upgrade = None
            best_score = -1.0
            for j in jobs:
                curr_model = models.get(mapping[j.job_id])
                if curr_model.name == FALLBACK:
                    continue
                for m in models:
                    delta_c = m.cost - curr_model.cost
                    if delta_c > 1e-9 and rem >= delta_c - 1e-9:
                        delta_q = m.quality(j.task) - curr_model.quality(j.task)
                        if delta_q > 0:
                            score = (delta_q / delta_c) * (1.0 + 0.1 * j.task.difficulty_rank)
                            if score > best_score:
                                best_score = score
                                best_upgrade = (j.job_id, m.name, delta_c)
            if best_upgrade is not None:
                jid, mname, delta_c = best_upgrade
                mapping[jid] = mname
                rem -= delta_c
            else:
                break

        return mapping

