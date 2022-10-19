# ML Project Documentation Template
---
## `1.` Overview
A summary of the doc's purpose, problem, solution, and desired outcome, usually in 3-5 sentences.

---

## `2.` Motivation
Why the problem is important to solve, and why now.

---

## `3.` Success metrics
Usually framed as business goals, such as increased customer engagement (e.g., CTR, DAU), revenue, or reduced cost.

---

## `4.` Requirements & Constraints
Functional requirements are those that should be met to ship the project. They should be described in terms of the customer perspective and benefit. (See this for more details.)
Non-functional/technical requirements are those that define system quality and how the system should be implemented. These include performance (throughput, latency, error rates), cost (infra cost, ops effort), security, data privacy, etc.
Constraints can come in the form of non-functional requirements (e.g., cost below $x a month, p99 latency < yms)
### `4.1` What's in-scope & out-of-scope?
Some problems are too big to solve all at once. Be clear about what's out of scope.

---

## `5.` Methodology
### `5.1` Problem statement
How will you frame the problem? For example, fraud detection can be framed as an unsupervised (outlier detection, graph cluster) or supervised problem (e.g., classification).
### `5.2` Data
What data will you use to train your model? What input data is needed during serving?
### `5.3` Techniques
What machine learning techniques will you use? How will you clean and prepare the data (e.g., excluding outliers) and create features?
### `5.4` Experimentation & Validation Results
How will you validate your approach offline? What offline evaluation metrics will you use?
If you're A/B testing, how will you assign treatment and control (e.g., customer vs. session-based) and what metrics will you measure?
### `5.5` Human-in-the-loop
How will you incorporate human intervention into your ML system (e.g., product/customer exclusion lists)?

---

## `6.` Implementation
### `6.1` High-level design
 
Start by providing a big-picture view. https://en.wikipedia.org/wiki/System_context_diagram and data-flow diagrams work well.
### `6.2.` Infra
How will you host your system? On-premise, cloud, or hybrid? This will define the rest of this section. 
### `6.3.` Performance (Throughput, Latency)
How will your system meet the throughput and latency requirements? Will it scale vertically or horizontally?
### `6.4.` Integration points
How will your system integrate with upstream data and downstream users? Endpoint APIs details with request response structures. 
### `6.5.` GitHub 
[Link to the Repository](https://github.com/jannikjw/one-shot-face-recognition)

---