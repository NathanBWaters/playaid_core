from playaid.pipeline import ID_TO_EMAILS
from playaid.postgres_connector import get_replays


if __name__ == "__main__":
    replays = get_replays()
    completed_replay_ids = set([r.replay_id for r in replays])
    incomplete_replay_ids_and_email = set(
        [
            (replay_id, email)
            for replay_id, email in list(ID_TO_EMAILS.items())
            if replay_id not in completed_replay_ids
        ]
    )
    print(
        "\n".join(
            [
                f'    "{replay_id}", // {email}'
                for replay_id, email in incomplete_replay_ids_and_email
            ]
        )
    )
