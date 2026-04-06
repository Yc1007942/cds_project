import { COOKIE_NAME } from "@shared/const";
import { getSessionCookieOptions } from "./_core/cookies";
import { systemRouter } from "./_core/systemRouter";
import { publicProcedure, router } from "./_core/trpc";
import { invokeOpenAI } from "./_core/openai";
import { AGENT_PERSONAS } from "../shared/agents";
import type { Message } from "./_core/openai";
import { z } from "zod";

export const appRouter = router({
    // if you need to use socket.io, read and register route in server/_core/index.ts, all api should start with '/api/' so that the gateway can route correctly
  system: systemRouter,
  auth: router({
    me: publicProcedure.query(opts => opts.ctx.user),
    logout: publicProcedure.mutation(({ ctx }) => {
      const cookieOptions = getSessionCookieOptions(ctx.req);
      ctx.res.clearCookie(COOKIE_NAME, { ...cookieOptions, maxAge: -1 });
      return {
        success: true,
      } as const;
    }),
  }),

  simulation: router({
    getAgentResponse: publicProcedure
      .input(
        z.object({
          agentId: z.string(),
          post: z.string().min(1).max(2000),
          conversationHistory: z.string(),
        })
      )
      .mutation(async ({ input }) => {
        const persona = AGENT_PERSONAS.find((a) => a.id === input.agentId);
        if (!persona) {
          throw new Error(`Agent not found: ${input.agentId}`);
        }

        const systemPrompt = `You are ${persona.name}, an AI agent in a social media simulation. ${persona.persona}

IMPORTANT RULES:
- Keep your response to 1-3 sentences maximum.
- Stay in character at all times.
- If there is conversation history, you may react to other agents' comments or the original post.
- Be concise and natural, like a real social media comment.`;

        const userPrompt = `A user posted the following:
"${input.post}"

${input.conversationHistory ? `Conversation so far:\n${input.conversationHistory}\n\n` : ""}Respond in character as ${persona.name}.`;

        const responseText = await invokeOpenAI([
          { role: "system", content: systemPrompt },
          { role: "user", content: userPrompt },
        ]);

        return {
          agentId: persona.id,
          agentName: persona.name,
          response: responseText.trim(),
        };
      }),
  }),
});

export type AppRouter = typeof appRouter;
