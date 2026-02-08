import { UnsupportedFunctionalityError } from '@ai-sdk/provider'
import { convertToBase64 } from '@ai-sdk/provider-utils'

import type { OpenAICompatibleChatPrompt } from './openai-compatible-api-types'
import type {
  LanguageModelV2Prompt,
  SharedV2ProviderMetadata,
} from '@ai-sdk/provider'

function getOpenAIMetadata(message: {
  providerOptions?: SharedV2ProviderMetadata
}) {
  return message?.providerOptions?.openaiCompatible ?? {}
}

export function convertToOpenAICompatibleChatMessages(
  prompt: LanguageModelV2Prompt,
): OpenAICompatibleChatPrompt {
  const messages: OpenAICompatibleChatPrompt = []
  for (const { role, content, ...message } of prompt) {
    const metadata = getOpenAIMetadata({ ...message })
    switch (role) {
      case 'system': {
        messages.push({ role: 'system', content, ...metadata })
        break
      }

      case 'user': {
        messages.push({
          role: 'user',
          content: content.map((part) => {
            const partMetadata = getOpenAIMetadata(part)
            switch (part.type) {
              case 'text': {
                return { type: 'text', text: part.text, ...partMetadata }
              }
              case 'file': {
                if (part.mediaType.startsWith('image/')) {
                  const mediaType =
                    part.mediaType === 'image/*' ? 'image/jpeg' : part.mediaType

                  return {
                    type: 'image_url',
                    image_url: {
                      url:
                        part.data instanceof URL
                          ? part.data.toString()
                          : `data:${mediaType};base64,${convertToBase64(part.data)}`,
                    },
                    ...partMetadata,
                  }
                } else {
                  throw new UnsupportedFunctionalityError({
                    functionality: `file part media type ${part.mediaType}`,
                  })
                }
              }
            }
          }),
          ...metadata,
        })

        break
      }

      case 'assistant': {
        let text = ''
        const toolCalls: Array<{
          id: string
          type: 'function'
          function: { name: string; arguments: string }
        }> = []

        for (const part of content) {
          const partMetadata = getOpenAIMetadata(part)
          switch (part.type) {
            case 'text': {
              text += part.text
              break
            }
            case 'tool-call': {
              toolCalls.push({
                id: part.toolCallId,
                type: 'function',
                function: {
                  name: part.toolName,
                  arguments: JSON.stringify(part.input),
                },
                ...partMetadata,
              })
              break
            }
          }
        }

        messages.push({
          role: 'assistant',
          content: text,
          tool_calls: toolCalls.length > 0 ? toolCalls : undefined,
          ...metadata,
        })

        break
      }

      case 'tool': {
        for (const toolResponse of content) {
          const output = toolResponse.output

          let contentValue: string
          switch (output.type) {
            case 'text':
            case 'error-text':
              contentValue = output.value
              break
            case 'content':
            case 'json':
            case 'error-json':
              contentValue = JSON.stringify(output.value)
              break
          }

          const toolResponseMetadata = getOpenAIMetadata(toolResponse)
          messages.push({
            role: 'tool',
            tool_call_id: toolResponse.toolCallId,
            content: contentValue,
            ...toolResponseMetadata,
          })
        }
        break
      }

      default: {
        const _exhaustiveCheck: never = role
        throw new Error(`Unsupported role: ${_exhaustiveCheck}`)
      }
    }
  }

  // Debug: dump OpenAI-format message summary to catch tool_use_id mismatches
  console.error('[SDK DEBUG] OpenAI-format messages (' + messages.length + '):')
  for (let i = 0; i < messages.length; i++) {
    const m = messages[i] as Record<string, unknown>
    const role = m.role as string
    if (role === 'tool') {
      console.error(`  [${i}] tool tool_call_id=${(m as { tool_call_id?: string }).tool_call_id}`)
    } else if (role === 'assistant') {
      const toolCalls = (m as { tool_calls?: Array<{ id: string; function?: { name: string } }> }).tool_calls
      if (toolCalls?.length) {
        const ids = toolCalls.map(tc => `${tc.function?.name}:${tc.id}`)
        console.error(`  [${i}] assistant tool_calls=[${ids.join(', ')}]`)
      } else {
        console.error(`  [${i}] assistant (text)`)
      }
    } else {
      console.error(`  [${i}] ${role}`)
    }
  }

  return messages
}
