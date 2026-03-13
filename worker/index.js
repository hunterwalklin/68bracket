/**
 * Cloudflare Worker — Chat proxy for 68bracket.
 *
 * Receives { context, messages } from the frontend, prepends bracket context
 * as a system prompt, and forwards to the Anthropic Messages API.
 *
 * Environment variables (set via `wrangler secret put`):
 *   ANTHROPIC_API_KEY  — your Anthropic API key
 *
 * wrangler.toml vars:
 *   ALLOWED_ORIGIN     — e.g. "https://hunterwalklin.github.io"
 */

export default {
  async fetch(request, env) {
    const origin = env.ALLOWED_ORIGIN || "*";

    // CORS preflight
    if (request.method === "OPTIONS") {
      return new Response(null, {
        headers: {
          "Access-Control-Allow-Origin": origin,
          "Access-Control-Allow-Methods": "POST, OPTIONS",
          "Access-Control-Allow-Headers": "Content-Type",
          "Access-Control-Max-Age": "86400",
        },
      });
    }

    if (request.method !== "POST") {
      return new Response("Method not allowed", { status: 405 });
    }

    try {
      const { context, messages } = await request.json();

      if (!messages || !Array.isArray(messages) || messages.length === 0) {
        return new Response(JSON.stringify({ error: "No messages" }), {
          status: 400,
          headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": origin },
        });
      }

      const systemPrompt = `You are the 68bracket assistant — a friendly, knowledgeable college basketball expert embedded in the 68bracket projected bracket website. Answer questions about the NCAA tournament bracket, seeds, matchups, bubble teams, and team stats.

Be concise (2-4 sentences unless the user asks for detail). Use the bracket data below as your source of truth for current projections.

${context || "No bracket data available."}`;

      const apiMessages = messages.slice(-10).map((m) => ({
        role: m.role,
        content: m.content,
      }));

      const resp = await fetch("https://api.anthropic.com/v1/messages", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          "x-api-key": env.ANTHROPIC_API_KEY,
          "anthropic-version": "2023-06-01",
        },
        body: JSON.stringify({
          model: "claude-haiku-4-5-20251001",
          max_tokens: 300,
          system: systemPrompt,
          messages: apiMessages,
        }),
      });

      if (!resp.ok) {
        const err = await resp.text();
        return new Response(JSON.stringify({ error: err }), {
          status: resp.status,
          headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": origin },
        });
      }

      const data = await resp.json();
      const reply = data.content?.[0]?.text || "No response";

      return new Response(JSON.stringify({ reply }), {
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": origin },
      });
    } catch (e) {
      return new Response(JSON.stringify({ error: e.message }), {
        status: 500,
        headers: { "Content-Type": "application/json", "Access-Control-Allow-Origin": origin },
      });
    }
  },
};
