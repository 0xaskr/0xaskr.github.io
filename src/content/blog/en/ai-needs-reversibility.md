---
title: 'Why DeepSeek Harness Chose Cordis'
description: 'When AI adds tools to its own runtime, components need lifecycle management, dependency coordination, and recovery. Cordis helps explain the design of DeepSeek Harness and the possibilities for personal harnesses and plugin platforms.'
pubDate: '2026/9/20'
draft: true
tags: ["Cordis", "AI", "Dynamic Composition", "Formal Verification", "Agent"]
---

Having AI write a new tool is no longer unusual. Could it go a step further: install that tool into its own running environment, remove it when it is no longer needed, and continue its other work along the way?

DeepSeek Harness already provides dedicated interfaces for this. A model can submit a plugin definition, then use tools to request that it be run, updated, stopped, or deleted. A plugin can extend capabilities on the host and include an interface in the browser. Definition and execution are separate, and execution follows the relevant lifecycle and approval procedures.

This gives AI-generated code another role: it can become a capability that the AI itself uses next.

The resulting questions are concrete. What happens if a service the new tool depends on is not ready? Who cleans up the event registrations and timers left by the old tool? If every adjustment requires a restart, how does work already in progress resume?

**Cordis addresses precisely these questions of components joining, leaving, and depending on one another at runtime.** In its [project introduction](https://github.com/deepseek-ai/deepseek-harness), DeepSeek Harness describes its architecture as “Everything is a Plugin” and explicitly states that it is built on Cordis. To understand that choice, we need to start with how a running environment is expected to change.

## How a Working Agent Changes Its Tools

A harness is the runtime environment around a model. It organizes model requests, provides tools, records sessions, manages tasks, and determines which operations require user approval. Through this environment, the model's capabilities reach files, terminals, and external services.

Suppose an agent is organizing project documentation. It queries documents through a retrieval tool, which in turn depends on an indexing service. Partway through the task, it discovers that the current index does not support a particular document format and prepares to replace the implementation.

Replacing the code file is only a small part of that work.

The old indexing service may still hold file watchers. The retrieval tool may still reference the old service. The new service needs time to initialize, and other tasks may also be querying the index. Simply inserting a new object leaves room for the old and new implementations to operate concurrently.

Developers handling this themselves must maintain startup ordering, deactivation notifications, cleanup callbacks, and reconnection logic across the application. As the number of components grows, each author has to understand part of someone else's lifecycle.

Cordis puts these relationships into a common component model. Components declare the services they need, provide capabilities to the environment, and let the runtime track the resources and cleanup operations they register. The relationships in our example become explicit: the indexing service provides a capability, the retrieval tool declares its dependency, and the runtime coordinates their activation and deactivation.

The two most useful concepts here are **temporal composability** and **spatial composability**. The names sound abstract, but each addresses an everyday problem.

### Temporal Composability: Undo Your Changes When You Leave

If an indexing component registers a file watcher, unloading it should remove that watcher. If it starts a scheduled task, unloading it should stop that task. Watchers and tasks registered by other components should remain intact.

Cordis expresses this relationship through revertible effects. An operation supplies a corresponding undo action, which the runtime records and arranges to execute when the component is deactivated. Resource acquisition and cleanup can therefore be written together, with a common mechanism managing the cleanup order.

“Undo” has a defined scope here. Removing a watcher undoes its registration; it cannot take back messages the watcher has already sent. Component authors still have to supply correct cleanup operations. The framework tracks and composes those operations; it cannot invent an inverse for arbitrary code.

### Spatial Composability: Adjust Together When Dependencies Change

A retrieval tool that needs an indexing service declares that dependency. The tool can become eligible for activation when the service is available. If the dependency becomes unavailable or resolves to a different provider, the runtime coordinates the tool's lifecycle accordingly.

This makes it easier to establish consistent conventions than having each component listen for “service available” and “service unavailable” events and maintain its own state. Developers can write components around what they need and leave the coordination of dependency changes to the runtime.

The two mechanisms work together. Dependency changes trigger deactivation; deactivation must undo registered resources before the component can potentially reactivate with new dependencies. The value of Cordis lies in bringing these actions into one lifecycle. The [paper](https://arxiv.org/abs/2608.25512) describes this relationship through effects, coeffects, and a calculus of dynamic composition.

## Why This Design Fits DeepSeek Harness

Cordis is a meta-framework for composing components. It provides contexts, services, events, and lifecycles on which applications can build their own functionality.

DeepSeek Harness takes this approach far. Its [architecture documentation](https://github.com/deepseek-ai/deepseek-harness/blob/master/docs/architecture.md) explains that model adapters, the tool registry, the session log, and even the agent loop itself are plugins. Extending a capability can therefore follow the same composition model that the rest of the application already uses.

For example, a separate plugin adds timeout policies to tool calls. It joins the tool execution pipeline, sets a deadline, and handles the result. Organizing the application this way lets a policy be installed, configured, and replaced independently, while making its point of intervention easier to inspect.

Plugins generated by the model can also enter this component system. The local installation I inspected was `@deepseek-ai/dsh@0.1.5-rc.2`, with Cordis version `4.0.2` actually installed. Its `dsh-tool-cordis` package provides these tools:

| Tool | Purpose |
|---|---|
| `cordis_define` | Submit a plugin definition and create a version ready to be run |
| `cordis_run` | Request activation, a version switch, or a rollback |
| `cordis_stop` | Stop the plugin's current run |
| `cordis_undefine` | Delete the plugin definition |

A successful definition does not mean the plugin has started. Unauthorized browser-side code, for example, must wait for approval, and asynchronous startup requires further observation of the result. These distinctions matter: the model proposes a modification, and the runtime incorporates it into the actual execution flow.

This is how I understand the choice of Cordis: **when the runtime itself consists of components, a model extending its capabilities has an established way to integrate them. When those components can be loaded, unloaded, and reconnected in an orderly way, the runtime has a foundation for ongoing adjustment.**

It is a technical choice consistent with the product's direction. The paper also identifies self-evolving agent harnesses as a target scenario. Still, the correspondence between features and design supports an architectural interpretation; it does not establish how DeepSeek actually reached its internal selection decision.

## Formalization Makes the Rules of Composition Inspectable

A component working correctly on its own does not necessarily keep working correctly when its loading and unloading interleave with those of other components. The indexing service might release its resources while the retrieval tool is still using it. Or one component's cleanup code might inadvertently remove another component's registration.

These problems lend themselves to explicit rules: when activation is allowed, when deactivation is required, in what order dependents and providers withdraw, and what conditions undo operations must satisfy.

An important contribution of the Cordis paper is to model these rules and prove properties such as temporal and spatial composability under explicit assumptions. This separates the obligations of individual components from the guarantees required of the composition mechanism.

That is one reason I find this work worth attention. With more components generated by AI, “how to integrate” and “how to withdraw” need meanings we can inspect. Otherwise, the time saved in generating code may be spent again on integration, debugging, and cleanup.

The benefit needs to be understood precisely, though. The paper proves properties of a calculus; the correspondence between the JavaScript implementation and that model also requires engineering checks. Section 5.1.1 explicitly states that the correctness of undo functions and the commutativity of relevant operations on a service are obligations of component providers. The runtime does not automatically verify these proof conditions. [Page 59 of the paper](https://arxiv.org/pdf/2608.25512#page=59)

Testing, code review, and formalization can therefore each do their part. Formalization helps establish the conditions under which composition rules hold. Tests check implementations and concrete scenarios. Review can uncover omissions in the contracts themselves. Adopting a framework with a formal model does not automatically give model-generated plugins a proof of correctness.

This also explains the reduction in complexity I hope for: consolidating lifecycle coordination scattered across components into shared mechanisms. Developers still have to write cleanup logic and validate business behavior, but they need not design a new dependency coordination protocol for every component. Whether this actually reduces defects and maintenance hours needs to be tested in real projects. There is no need to invoke “entropy reduction” in place of that engineering accounting.

## Why Component Unloading Still Leaves Restart and Recovery to Solve

Return to the document retrieval example. If only the indexing service needs replacing, component lifecycle management may keep the impact within that service and its dependents. Unrelated functionality need not be rebuilt solely because of this replacement.

How an ongoing query finishes, or how data in the old index migrates, still requires a concrete design. Being able to load and unload components does not mean in-flight requests and application state transfer automatically without loss.

A process restart raises another set of questions. A session log can restore recorded conversations and tool results, but it cannot directly restore a network request waiting for a response or an unfinished asynchronous call.

DSH's recovery logic makes a specific distinction here. If a tool call has been recorded as started but has no durable result, session recovery marks its outcome as unknown. For an operation that may have side effects, external state needs to be checked before deciding whether to retry. A request may already have modified remote data even though its result never made it into the log.

Dynamic plugins illustrate the same distinction. In the local version described above, `dsh-cordis-host-runner` stores its dynamic registry in an in-process `Map`. Recreating the process does not automatically preserve that registry, its running instances, or its live handles. However, plugin source code may still appear in persisted tool-call arguments, so it would be wrong to conclude that all the code the model wrote has disappeared.

A harness intended to work for long periods therefore needs both component loading and unloading, and persistence, state migration, and failure recovery. These address ordinary adjustments and continuation after interruptions, respectively. Cordis's composition mechanisms form one part of that engineering work.

## A Plugin Hub Can Let the Platform Handle the Complexity

If everyone has to inspect every piece of plugin code a model generates, a personalized harness will struggle to become a mass-market product. A Hub that distributes, reviews, and maintains plugins has a natural role to play.

My proposed approach gives expert and general users different ways to participate.

Expert users can develop their own plugins, adjust component combinations, and experiment in environments with explicit permissions. General users can prefer plugins and combinations maintained by the Hub, delegating version selection, compatibility validation, and update review to the platform. Experts' accumulated experience can also enter the shared ecosystem as plugins or complete configurations.

**Personal evolution and centralized review can coexist.** Different users needing different combinations of tools does not mean every user must independently maintain the entire software stack.

A platform can do much more than glance at a description and approve it. It can inspect source code and dependencies, maintain test environments, validate supported component combinations, and record review results for each version. Requiring every update to go through review helps prevent users from unknowingly moving from a reviewed version to an unchecked implementation.

In such a product, trusting the Hub by default is an engineering decision to delegate trust. The platform needs to explain what it has validated and which combinations it supports; the runtime needs to enforce the corresponding permission and version policies. If users mix in components they have modified themselves, the interface should make clear whether their environment remains within the platform's validated scope.

Capability restrictions can also follow DSH's plugin architecture. For example, the tool execution pipeline can check paths or parameters, and restricted service implementations can provide file and network access. Whether these restrictions hold depends on the actual execution path: if code can bypass the restricted interfaces, a policy plugin alone does not establish a complete security boundary. DSH's own [safety notice](https://github.com/deepseek-ai/deepseek-harness/blob/master/SAFETY.md) likewise does not describe its existing sandboxing and approvals as guarantees of isolation.

There need not be one prescribed way to handle plugins the model generates on the spot. Some scenarios suit an initial trial in a restricted environment. Others allow quick approval through automated checks, while still others should wait for human review. A platform can choose different procedures according to capabilities and risk, then decide which results are suitable for preservation, distribution, and long-term maintenance.

Such a Hub can centralize source review, compatibility maintenance, and incident response, reducing the work that general users have to handle alone. That work gives the platform a concrete way to demonstrate its value.

## Everyone's Harness May Gradually Become Different

This readily brings browsers to mind: people use the same underlying software and open ecosystem, yet develop their own working environments through different extensions, settings, and habits.

Harnesses may evolve in a similar way. Researchers accumulate components for literature search and experiment management. Developers accumulate code-checking and deployment tools. Designers accumulate workflows for asset processing and previews. The same model, placed among different tools, memories, and working conventions, can produce very different experiences.

If AI can help write and adjust these components, customizing an environment may become easier still. A user describes a need; the agent selects an existing plugin or generates an implementation, then integrates it into the runtime. Effective approaches are retained and become available for others to reuse.

The coordination required here is concrete. Cordis manages component composition and lifecycles, the Harness manages models and tasks, and the Hub helps distribute and maintain capabilities that have been checked. Experience can accumulate in memory, tools, and workflows. Continual learning at the model level can also interact with these changes, though each has its own validation and recovery mechanisms.

This remains a vision of the future. How far it can go depends on whether the generated components are useful, whether their maintenance is affordable, and whether users can understand and control their runtime environments.

This is where I think Cordis is easy to underestimate: it takes adding features to software a step further, toward managing ongoing changes in the software's capabilities. As AI becomes both a user and an author of tools, this foundation will deserve increasingly close attention.

---

The runtime details in this article were checked on September 20, 2026, against local installations of `@deepseek-ai/dsh@0.1.5-rc.2` and `@deepseek-ai/cordis@4.0.2`. The inspection covered package metadata and compiled artifacts only; it did not include process-restart experiments or isolation tests with malicious plugins. Tool definitions, the dynamic registry, and interruption recovery are located in `dsh-tool-cordis/lib/index.js`, `dsh-cordis-host-runner/lib/index.js`, and `dsh-session/lib/index.js`, respectively. The timeout policy example is based on `dsh-tool-call-timeout-policy/lib/index.js`. These observations are specific to that version; later implementations may change.
