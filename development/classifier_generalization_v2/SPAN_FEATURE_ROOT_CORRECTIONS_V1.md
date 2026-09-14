# Root corrections following independent feature review

The original numerical dimensions were correct in executable arithmetic, but the source comment, handoff and earlier coordination incorrectly totalled them as10224. Correct sum:3072+3072+2048+36+44=8272. This was a documentation arithmetic mistake, not extra implemented features. The historical handoff/review remain unchanged.

The real transform bug was converting a ragged list of token windows to one NumPy array before validation. Root now recognizes a bare3DNumPywindow explicitly and otherwise validates each batch member separately, allowing different1..16token lengths withoutpadding.

Added regressions for mixed-length training plus batch transformation, mixed-length validation, and an explicit8272dimension assertion. All21synthetic tests PASS (0.449seconds). No realdata, Qwenforwards or classifierfits occurred. Independent re-review is requested; initial BLOCKING review is preserved.

NewsourceSHA256:a23aaa38dc36a764af8cd30d2d8ed40193a9a3496a9d8687778c96f74741c5c0.
NewtestSHA256:76117a26cd7b80406761ce13c23ad469c99012dec8a3a4e706ec1e9a5cc19c48.
