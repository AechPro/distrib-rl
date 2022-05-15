from trueskill import rate_1vs1, Rating
from distrib_rl.Distrib import RedisKeys
from distrib_rl.Utils import CompressionSerialisation as cser
from redis import Redis
import time
import os


class OpponentSelector(object):
    def __init__(self, cfg):
        self.cfg = cfg
        self.rng = cfg["rng"]
        self.player_skills = []
        self.policy_skill = Rating(100)
        self.known_policies = []

        ip = os.environ.get("REDIS_HOST", default='localhost')
        port = os.environ.get("REDIS_PORT", default=6379)
        password = os.environ.get("REDIS_PASSWORD", default=None)

        self.redis = Redis(host=ip, port=port, password=password)

    def get_opponent(self):
        encoded = self.redis.get(RedisKeys.MARL_CURRENT_OPPONENT_KEY)
        if encoded is None:
            return None, -1

        decoded = cser.unpack(encoded)
        return decoded

    def update_opponent(self):
        if self.cfg["rng"].randint(0, 10) > 2 or len(self.player_skills) == 0:
            encoded = cser.pack((-1, -1))
            self.redis.set(RedisKeys.MARL_CURRENT_OPPONENT_KEY, encoded)
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

        encoded = cser.pack((params, opponent_num))
        self.redis.set(RedisKeys.MARL_CURRENT_OPPONENT_KEY, encoded)

    def update_ratings(self):
        red = self.redis
        key = RedisKeys.MARL_MATCH_RESULTS_KEY
        current = red.lpop(key)
        results = []

        while current is not None:
            decoded = cser.unpack(current)
            current = red.lpop(key)
            if decoded[0] != -1:
                results.append(decoded)

        for result in results:
            opponent_num, victory = result
            if opponent_num >= len(self.player_skills):
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
        success = False
        while not success:
            try:
                encoded = cser.pack((opponent_num, victory))
                success = True
            except MemoryError:
                print("Failed to submit MARL result...")
                time.sleep(1)

        self.redis.lpush(RedisKeys.MARL_MATCH_RESULTS_KEY, encoded)
