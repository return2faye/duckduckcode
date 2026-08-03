FACTS = {
    37: "AUTHORITATIVE mode=strict",
    241: "AUTHORITATIVE retry_limit=4",
    479: "AUTHORITATIVE timeout_ms=2750",
}

for index in range(520):
    print(
        FACTS.get(
            index,
            f"NOISE record={index:04d} region=west checksum={index * 7919 % 100000:05d} "
            f"candidate_mode=legacy candidate_retry={index % 9}",
        )
    )
