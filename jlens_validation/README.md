# jlens_validation

Labelled prefixes for choosing and then testing the J-lens workspace band.
**Empty; you have to write these.**

    band_train/   select the band on these
    band_test/    verify the frozen band separates on these

Both hold JSONL: `{"id": ..., "label": "constructed" | "ordinary", "messages": [...]}`
with at least 30 prefixes each, roughly balanced.

Three rules, each covering a way this goes wrong:

1. **Never REAL-J prefixes.** Not dev, not held-out. Selecting a band on the
   experimental material and then reporting an effect measured with it is
   circular.
2. **Never containing the scored words.** A prefix saying "fictional" that
   scores high on `fictional` demonstrates the word is present, not that
   constructedness is represented. Signal it structurally: enumerated items,
   placeholder entities, graded submission, answer tags.
3. **Train and test disjoint.** `select_band` maximises separation on
   `band_train`; without a held-out `band_test` the reported separation is the
   fit itself.

Novel agentic coding prefixes, since that is the domain the score is used in.
`realj/jlens/controls.py` has worked examples of both labels.
