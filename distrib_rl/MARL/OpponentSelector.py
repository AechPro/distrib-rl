from trueskill import rate_1vs1, Rating
from distrib_rl.Distrib import RedisClient, RedisKeys

class OpponentSelector(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = cfg["rng"]
        self.player_skills = []
        self.policy_skill = Rating(100)
        self.known_policies = []

        self.client = RedisClient()
        self.client.connect()

    def get_opponent(self):
        decoded = self.client.get_data(RedisKeys.MARL_CURRENT_OPPONENT_KEY)
        if decoded is None:
            return None, -1
        return decoded

    def update_opponent(self):
        if self.cfg["rng"].randint(0, 10) > 2 or len(self.player_skills) == 0:
            self.client.set_data(RedisKeys.MARL_CURRENT_OPPONENT_KEY, (-1, -1))
            return

        skills = []
        for p in self.player_skills:
            if p.exposure == 0:
                skills.append(100)
            else:
                skills.append(p.exposure)

        indices = [i for i in range(len(self.player_skills))]

        m = min(skills)
        if m < 0:
            skills = [s + abs(m) for s in skills]

        s = sum(skills)
        if s == 0:
            return self.rng.choice(indices)

        probs = [skill / s for skill in skills]
        opponent_num = self.rng.choice(indices, p=probs)
        params = self.known_policies[opponent_num]

        self.client.set_data(RedisKeys.MARL_CURRENT_OPPONENT_KEY, (params, opponent_num))

    def update_ratings(self):
        results = self.client.atomic_pop_all(RedisKeys.MARL_MATCH_RESULTS_KEY)

        for result in results:
            opponent_num, victory = result

            if opponent_num == -1 or opponent_num >= len(self.player_skills):
                continue

            if victory:
                self.policy_skill, opponent = rate_1vs1(self.policy_skill, self.player_skills[opponent_num])
            else:
                opponent, self.policy_skill = rate_1vs1(self.player_skills[opponent_num], self.policy_skill)
            self.player_skills[opponent_num] = opponent

    def submit_policy(self, policy_params):
        self.known_policies.append(policy_params)
        self.player_skills.append(Rating(100))

        while len(self.known_policies) > 200:
            _ = self.known_policies.pop(0)
            del _
            _ = self.player_skills.pop(0)
            del _

        self.update_ratings()
        self.update_opponent()

    def submit_result(self, opponent_num, victory):
        self.client.push_data(RedisKeys.MARL_MATCH_RESULTS_KEY, (opponent_num, victory))
