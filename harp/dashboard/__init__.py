"""Developer dashboard — observe the running agent (not part of the user flow).

A web view for us, not the visitor: live transcripts, retrieved context, current
face-ID, latency, and agent health. It is a pure OBSERVER — it subscribes to the
bus and renders; it must never publish control events or drive behavior. Built
late (see PLAN.md phase 8).
"""
