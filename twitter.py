#!/usr/bin/env python3
from helpers.twitter_helpers import post_all_unposted, get_x_metrics


if __name__ == "__main__":
    # resp = post_all_unposted()
    stats = get_x_metrics("1990176746616885405")
    print(stats)