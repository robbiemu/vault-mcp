# Code Deconstructed
**Episode 12: Vault-MCP – Giving Your AI Assistant the Company Wiki**

---

**[INTRO MUSIC: 5 seconds of upbeat, curious synth-pop fades under]**

**HOST (Priya):**  
Hello and welcome back to *Code Deconstructed*, the show where we clone the repo, pop open the `README`, and get the real story behind the code. I'm your host, Priya.

Today, we're diving into a fascinating project that solves a problem I think every developer with an AI assistant has felt. It's a small server with a huge mission called **Vault-MCP**, and its goal is to let your AI coding partner read your team's documentation—the project plans, the architectural notes, the "here be dragons" warnings—without you having to copy-paste a novel into the context window.

To help us unpack it, I'm joined by **Lucas**, a developer who's been working with this project. Lucas, welcome to the show!

**LUCAS (technical, matter-of-fact, slightly dry humor):**  
Hey Priya. Thanks for having me. Fair warning - I've had too much coffee, so I might get a bit into the weeds on the implementation details.

---

### Segment 1 – The Real-World Problem

**PRIYA:**  
(Laughs) Perfect, that's exactly what we want. So, let's set the stage. I'm a developer, I'm using an AI tool like Cursor or maybe the new Crush CLI that's been making waves, and it's brilliant. It can refactor my code, write unit tests, it understands the codebase itself. But it has absolutely no idea about the 200-page Obsidian vault where my team keeps everything else: the high-level architecture diagrams, the scope for the current epic, and most importantly maybe with tasks, like the risk register that basically says, "The new-checkout-flow feature flag must be disabled in production." Or, "Performance Note: The user profile query is not optimized."

If I paste that whole vault into the chat, the agent's context window just gives up. But if I paste nothing, it will happily refactor the billing module into a beautiful, modern, and completely non-functional pile of code, and then we all cry.

This is the exact gap Vault-MCP is built to fill, right?

**LUCAS:**  
Yeah, that's basically it. The problem is you've got this context mismatch - the AI knows your code but not your institutional knowledge. Vault-MCP sits in the middle and does semantic search over your docs. When the AI asks a question, it doesn't get your entire knowledge base dumped on it. Instead, it gets the three or four most relevant chunks, which keeps you under token limits while still giving useful context.

---

### Segment 2 – The Feature Tour

**PRIYA:**  
Alright, let's talk about what it actually does. What are the core features?

**LUCAS:**  
The main thing is it indexes Markdown files from any folder structure. It handles standard Markdown, but it also knows about Obsidian-specific stuff like wiki links and Joplin notebook exports.

**PRIYA:**  
What about when you don't want everything indexed? Like, I don't need my grocery list showing up in technical discussions.

**LUCAS:**  
Right, there's prefix filtering. You configure it to only look at files that start with certain prefixes. So if your work docs all start with "Project-" or whatever, it'll ignore everything else.

**PRIYA:**  
And when I update a document?

**LUCAS:**  
It has a file watcher that picks up changes automatically. Usually takes a couple seconds to re-index after you save something.

**PRIYA:**  
Now, you mentioned it has both REST and MCP interfaces, and two different retrieval modes?

**LUCAS:**  
Yeah, so there's a REST API if you want to query it directly or build custom tooling. And there's an MCP server for AI agents that understand that protocol. For retrieval, you can run it in static mode where it just returns the raw text chunks it finds, or agentic mode where it uses an LLM to rewrite those chunks to be more directly responsive to the question.

---

### Segment 3 – A Stroll Through the Commit History

**PRIYA:**  
I love looking at how projects evolve. The changelog shows this went through some pretty significant architectural changes.

**LUCAS:**  
Oh yeah, it's been quite a journey. Version 0.1 was basically just proving you could wire FastAPI and ChromaDB together. It worked, but that was about it.

**PRIYA:**  
Version 0.2 seems like where things got more ambitious?

**LUCAS:**  
That's when we tried to do everything at once. Rewrote it to use LlamaIndex, planned out this whole agentic system... but most of the advanced features were just stubs. Classic over-engineering problem - looked great on paper, but half of it was just TODO comments.

**PRIYA:**  
I think every developer has written that version.

**LUCAS:**  
Exactly. Version 0.3 was about actually implementing the agent stuff - that's when the LLM-powered chunk rewriting actually started working. And 0.4 was a big refactor to clean up the architecture. Split everything into a core VaultService with separate server layers for REST and MCP. Much cleaner separation of concerns.

---

### Segment 4 – The Architectural Deep Dive

**PRIYA:**  
Let's get technical. You made some interesting optimization choices.

#### a) Filter-Then-Load

**LUCAS:**  
Yeah, the original approach was pretty naive. It would load every single Markdown file into memory first, then check if the filename matched the filter criteria. On a vault with a few thousand files, that meant 30-second startup times just to throw away most of what you loaded.

**PRIYA:**  
So what's the better approach?

**LUCAS:**  
Just scan the filenames first - that's fast - build a list of what matches your filters, then only load those files. Cut startup time from 30 seconds to about 3.

#### b) Merkle-Tree Sync

**PRIYA:**  
The changelog mentions Merkle tree sync for efficiency. That sounds very Git-like.

**LUCAS:**  
It's the same concept, yeah. The goal is to avoid re-reading every file just to see if one has changed. For every file we're tracking, we compute a hash of its contents—a unique signature for that version of the file.

The Merkle tree then takes all of those individual file hashes and recursively hashes them together until you get one single, top-level 'root hash'. That one hash now represents the precise state of all your tracked documents. When a file is saved, we just recalculate the root hash. If it's different from the one we have stored, we know something changed and can quickly pinpoint which files to re-process. If it's the same, we do nothing. It's incredibly fast.

#### c) Static vs. Agentic Retrieval

**PRIYA:**  
Break down the difference between static and agentic modes for me.

**LUCAS:**  
Sure. Let's say you ask about deployment risks for the billing module.

Static mode finds the relevant document section and gives you the raw text - headings, bullet points, whatever was in the original markdown. It's fast and deterministic.

Agentic mode finds the same content, but then runs it through an LLM to rewrite it as a direct response to your specific question. So instead of getting a generic section titled "Deployment Considerations," you might get something that starts with "The primary deployment risk for the billing module is database migration conflicts" - it's been transformed to directly answer what you asked.

---

### Segment 5 – Peeking at the Future

**PRIYA:**  
What's on the roadmap? Any features you're excited about?

**LUCAS:**  
There are a couple of things that would be interesting to implement. One is answer synthesis - right now even agentic mode gives you multiple chunks that you have to read through. It would be cool to have an optional final step that takes those chunks and synthesizes them into a single coherent answer.

**PRIYA:**  
Like completing the RAG pipeline with actual generation.

**LUCAS:**  
Exactly. The other thing is making the agent's toolkit more configurable. We've already moved the system prompts into a .toml file, so users can easily tweak the agent's persona and instructions. But the tools it can use—the actual functions like read_full_document—are still defined in Python. It would be neat if you could define new, custom tools entirely in a config file, allowing the agent to learn new tricks without anyone touching the core code.

---

### Segment 6 – Try It Yourself

**PRIYA:**  
Where can people try this out?

**LUCAS:**  
It's on GitHub. If you have uv installed - that's the Python package and project manager - you can clone the repo and get it running pretty quickly. Main thing you need to do is edit the config file to point it at your documents folder. If you're using an MCP-compatible client, you just point it at the server and can start querying your docs.

---

**[OUTRO MUSIC starts soft, building gently]**

**PRIYA:**  
And that's Vault-MCP—a project that turns your knowledge base into a context-aware assistant for your AI tools. Thanks to Lucas for walking us through the technical details.

**LUCAS:**  
Thanks for having me. Now I should probably go fix those TODO comments I mentioned earlier.

**PRIYA:**  
(Chuckles) The work never ends. And to our listeners: check out the repo, try it with your own docs, and remember—good documentation deserves good tooling. We'll see you next time on *Code Deconstructed*!

**[MUSIC swells to finish]**