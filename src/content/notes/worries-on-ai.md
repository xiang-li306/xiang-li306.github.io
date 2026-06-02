A small detour: I just want to express some of my personal opinions.

As we all know, AI has become extremely popular in recent years, especially language models. Millions of researchers and engineers are pouring into this area, training models, designing new skills, and building more and more applications around LLMs. But actually, as a machine learning theory researcher, I have been wondering and worrying: this is not exactly the scenario I expected to see.

Currently, almost everyone is training language models under the same broad paradigm: pre-training, post-training, and deployment. Most of these systems are still based on the transformer architecture. Even new things like vision models and agents are often built around transformer-based language models. This raises a question that I cannot stop thinking about: does a language model, or more specifically a transformer, really represent intelligence?

To some extent, I think yes. But when I was an undergraduate, I took courses like Introduction to AI, which introduced AI from 30 or 40 years ago. I read books such as Artificial Intelligence: A Modern Approach, and I was fascinated by how many researchers from different disciplines were trying to understand intelligence. There were ideas from statistics, computer science, mathematics, psychology, neuroscience, logic, and many other areas. Many old ideas in search, planning, logic, and learning were beautiful to me.

![A researcher looking at the branching history of AI ideas](/notes/worries-on-ai/research-paradigm.png)

But now, language models do not seem to carry that same kind of beauty, at least to me. Why? Because after we entered the deep learning era, it often feels like better performance comes from a larger computation budget, more GPUs, more data, and more parameters. The scaling law is surprising to everyone, including me. But somehow, for researchers like me and for many theorists, it is also counterintuitive. In biology, intelligence seems to be gradually evolved, rather than simply "emerged" after crossing a certain scale.

Now everyone wants to earn money, and many people are pouring into AI companies in industry. I understand that. But I still think fundamental research, and academia more broadly, are extremely important. As AI researchers, we should not be constrained by one fixed architecture or one fixed language-model paradigm.

At least, I see several problems.

## Data-Driven Intelligence and Exact Reasoning

Current language models are almost entirely data-driven, or more precisely, statistical. The next-token prediction paradigm is based on estimating how probable the next token should be. But that is possibly not the case in biology, or at least not in carbon-based biology.

In many problems, such as mathematics or logic, we need exact reasoning rather than a "probably approximately correct" style of reasoning. In the early stages of AI, many agents were based on formal languages or logic. Under a given set of rules, such systems could be absolutely correct. Of course, they were limited in many ways, but their internal structure was explicit and interpretable.

![Probability streams contrasted with formal reasoning structures](/notes/worries-on-ai/probability-and-logic.png)

Language models, by contrast, often behave like extremely powerful statistical machines. They can produce impressive answers, but the mechanism is not the same as deriving a conclusion from rules. This does not mean they are useless for reasoning. Actually, they are surprisingly useful. But I still worry that if we treat statistical prediction as the whole story of intelligence, we may miss something important.

## Knowledge Cutoffs and Self-Evolution

Current language models also have knowledge cutoffs. This means they are trained with data collected up to a fixed date. They can search the Internet to get more current knowledge and put it into context, but that still does not feel fully "intelligence-like" to me.

I think intelligence should be able to self-evolve, rather than being retrained every several months. Future AI systems, in my opinion, should definitely become more self-evolving. They should not be constrained forever by a human-designed transformer architecture and the current training paradigm.

But then another question appears: what level is human intelligence? Currently, the knowledge width of LLMs definitely surpasses humans. They know something about almost every field. But their knowledge depth is still more complicated to judge. In the future, silicon-based intelligence, which we call AI, may surpass human intelligence much more completely. If that happens, continuing to use only human-designed architectures may constrain its evolution. Perhaps such systems should be able to explore the most suitable structure for themselves.

## Carbon-Based and Silicon-Based Intelligence

We can also compare silicon-based intelligence, such as a language model, with carbon-based intelligence, such as a human.

Oh wait. Writing to this point makes me wonder again: what is intelligence? At least I would say it includes the ability to learn, communicate, reason, adapt, and interact with the world. But here I want to temporarily treat intelligence as a physical object. If we do that, one important aspect is energy consumption.

As humans, we can eat a simple breakfast and stay energetic for a whole day. But large language models require a huge amount of electricity, especially during training and large-scale inference. I do not want to make an overly precise claim here, because the numbers depend on the model, hardware, data center, and workload. Still, the contrast is important: biological intelligence is extremely energy-efficient, while current silicon-based intelligence depends on massive infrastructure.

![A quiet comparison between biological energy and computational infrastructure](/notes/worries-on-ai/energy-gap.png)

For now, humans can still cut off the electricity source for LLMs and make them "dead." But what if, in the future, AI systems become more distributed, more autonomous, and more deeply connected to infrastructure? This is not only a technical question. It is also a social and philosophical one.

## Weak Intelligence and Strong Intelligence

This discussion gradually steps into a deeper question. I still want to compare the notions of weak intelligence and strong intelligence.

Although LLMs can absorb and use a huge amount of human knowledge, I still think they are closer to weak intelligence, because they rely heavily on language. But the world has many more aspects: vision, audio, physical interaction, embodiment, environment, causality, and so on. The world is much richer than what language can fully describe. Actually, it is much richer than what humans can observe too. The same story applies to AI.

So somehow I think that if one day AI can "master the world," it must be a form of strong intelligence that reaches every aspect of the world, not only language. This, I believe, still needs time.

Many AI researchers, including me, believe that the next generation of AI, or the next huge breakthrough, may come in the next 5 to 20 years. But I think strong intelligence may need much longer. More importantly, I think future breakthroughs should force us to rethink the intrinsic aspects of intelligence, rather than simply continuing the current direction of scaling.

And in that process, academia should still play an important role. Industry can move fast and build powerful systems, but fundamental research asks slower and deeper questions. What is intelligence? What should be learned? What should be reasoned? What should be evolved? What kind of architecture is not only useful, but also conceptually right?

These are my worries on AI.
