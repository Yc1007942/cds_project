import { ENV } from "./env";

export type Message = {
  role: "system" | "user" | "assistant";
  content: string;
};

export type OpenAIResponse = {
  id: string;
  object: string;
  created: number;
  model: string;
  choices: Array<{
    index: number;
    message: {
      role: string;
      content: string;
    };
    finish_reason: string;
  }>;
  usage: {
    prompt_tokens: number;
    completion_tokens: number;
    total_tokens: number;
  };
};

export async function invokeOpenAI(
  messages: Message[],
  model: string = "gpt-4o-mini"
): Promise<string> {
  if (!ENV.openaiApiKey) {
    throw new Error("OPENAI_API_KEY is not configured");
  }

  const response = await fetch("https://api.openai.com/v1/chat/completions", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
      Authorization: `Bearer ${ENV.openaiApiKey}`,
    },
    body: JSON.stringify({
      model,
      messages,
      temperature: 0.7,
      max_tokens: 500,
    }),
  });

  if (!response.ok) {
    const errorText = await response.text();
    throw new Error(
      `OpenAI API failed: ${response.status} ${response.statusText} – ${errorText}`
    );
  }

  const data = (await response.json()) as OpenAIResponse;
  const content = data.choices[0]?.message?.content;

  if (!content) {
    throw new Error("No response content from OpenAI API");
  }

  return content;
}
