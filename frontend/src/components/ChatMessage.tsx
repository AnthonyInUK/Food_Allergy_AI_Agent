'use client'

import React, { useEffect, useMemo, useState } from 'react'
import ReactMarkdown from 'react-markdown'
import { ChevronDown, Sparkles, Zap } from 'lucide-react'
import type { Message, ReasoningStep } from '@/lib/api'

interface ChatMessageProps {
    message: Message
}

/** 旧版后端把「查询路径」写在同一条 Markdown 里时的解析 */
function parseLegacyQueryBlock(text: string): {
    bodyMarkdown: string
    steps: ReasoningStep[]
    totalSeconds?: number
} | null {
    const pathRx = /\n+\*\*查询路径[：:]\*\*\s*\n?|\n+查询路径[：:]\s*\n?/i
    const m = text.match(pathRx)
    if (!m || m.index === undefined) return null

    const head = text.slice(0, m.index).trim()
    let tail = text.slice(m.index + m[0].length).trim()
    if (!tail) return null

    let totalSeconds: number | undefined
    const timeRx = /\n+\*\*耗时[：:]\*\*\s*([\d.]+)\s*s?|\n+耗时[：:]\s*([\d.]+)\s*s?|\*\*耗时[：:]\*\*\s*([\d.]+)\s*s?|耗时[：:]\s*([\d.]+)\s*s?/i
    const tm = tail.match(timeRx)
    if (tm) {
        const num = tm[1] || tm[2] || tm[3] || tm[4]
        if (num) totalSeconds = parseFloat(num)
        tail = tail.slice(0, tm.index).trim()
    }

    const segments = tail
        .split(/→/)
        .map((s) => s.trim())
        .filter((s) => s.length > 0 && !/^查询路径[：:]?$/i.test(s))

    if (segments.length === 0) return null

    const steps: ReasoningStep[] = segments.map((seg, i) => {
        const dur = seg.match(/\(([\d.]+)\s*s\)\s*$/)
        if (dur) {
            const label = seg.replace(/\s*\([\d.]+\s*s\)\s*$/, '').trim()
            return { key: `legacy-${i}`, label, seconds: parseFloat(dur[1]) }
        }
        return { key: `legacy-${i}`, label: seg }
    })

    return { bodyMarkdown: head, steps, totalSeconds }
}

function ReasoningDetails({
    steps,
    totalSeconds,
    cached,
    streaming,
    partialTotal,
}: {
    steps: ReasoningStep[]
    totalSeconds?: number
    cached?: boolean
    streaming?: boolean
    partialTotal?: number
}) {
    const [isOpen, setIsOpen] = useState<boolean>(!!streaming)

    useEffect(() => {
        if (streaming) setIsOpen(true)
    }, [streaming])
    if (!steps.length && !streaming) return null

    const timeLabel =
        streaming && typeof partialTotal === 'number' && !Number.isNaN(partialTotal)
            ? `已用 ${partialTotal.toFixed(2)}s`
            : typeof totalSeconds === 'number' && !Number.isNaN(totalSeconds)
              ? `共 ${totalSeconds.toFixed(2)}s`
              : null

    return (
        <details
            className="mb-3 rounded-xl border border-slate-200/90 bg-slate-50/95 text-slate-700 shadow-sm group/trace"
            open={isOpen}
            onToggle={(e) => {
                const open = (e.currentTarget as HTMLDetailsElement).open
                setIsOpen(open)
            }}
        >
            <summary className="flex cursor-pointer list-none items-center gap-2 px-3 py-2.5 text-xs select-none [&::-webkit-details-marker]:hidden">
                <ChevronDown
                    size={16}
                    className="shrink-0 text-slate-500 transition-transform group-open/trace:rotate-180"
                    aria-hidden
                />
                <Sparkles size={14} className="shrink-0 text-violet-500" aria-hidden />
                <span className="font-medium text-slate-600">
                    思考过程
                    {streaming ? <span className="ml-1 text-violet-600 font-normal">· 进行中</span> : null}
                </span>
                {cached ? (
                    <span className="ml-1 inline-flex items-center gap-0.5 rounded-full bg-amber-100 px-1.5 py-0.5 text-[10px] font-medium text-amber-800">
                        <Zap size={10} />
                        缓存命中
                    </span>
                ) : null}
                {timeLabel ? <span className="ml-auto tabular-nums text-slate-400">{timeLabel}</span> : null}
            </summary>
            <div className="border-t border-slate-200/80 bg-white/70 px-3 py-2">
                {!steps.length && streaming ? (
                    <p className="text-xs text-slate-500 py-1 animate-pulse">正在连接模型与检索数据…</p>
                ) : null}
                <ol className="space-y-2">
                    {steps.map((step, i) => (
                        <li
                            key={step.key || `${i}`}
                            className="relative border-l-2 border-violet-200 pl-3 text-xs leading-relaxed text-slate-600"
                        >
                            <span className="font-medium text-slate-700">{step.label}</span>
                            {(() => {
                                const key = (step.key || '').toLowerCase()
                                const label = (step.label || '').toLowerCase()
                                let tag = ''
                                if (key.includes('retrieve') || label.includes('检索') || label.includes('retrieval')) {
                                    tag = '检索'
                                } else if (key.includes('generate') || label.includes('生成') || label.includes('generate')) {
                                    tag = '生成'
                                } else if (
                                    key.includes('vision') ||
                                    label.includes('图片') ||
                                    label.includes('vision') ||
                                    label.includes('识别')
                                ) {
                                    tag = '视觉'
                                } else if (key.includes('translate') || label.includes('翻译') || label.includes('translate')) {
                                    tag = '翻译'
                                } else if (
                                    key.includes('database') ||
                                    key.includes('sql') ||
                                    label.includes('数据库') ||
                                    label.includes('sql')
                                ) {
                                    tag = '数据库'
                                } else if (key.includes('web') || label.includes('联网') || label.includes('web')) {
                                    tag = '联网'
                                } else if (key.includes('plan_route') || label.includes('路由规划')) {
                                    tag = '路由'
                                }
                                return tag ? (
                                    <span className="ml-2 inline-flex items-center rounded-full bg-slate-200 px-1.5 py-0.5 text-[10px] text-slate-600">
                                        {tag}
                                    </span>
                                ) : null
                            })()}
                            {typeof step.seconds === 'number' && !Number.isNaN(step.seconds) ? (
                                <span className="ml-2 tabular-nums text-slate-400">{step.seconds.toFixed(2)}s</span>
                            ) : null}
                            {step.detail ? (
                                <p className="mt-0.5 text-[11px] text-slate-500">{step.detail}</p>
                            ) : null}
                        </li>
                    ))}
                </ol>
            </div>
        </details>
    )
}

export const ChatMessage: React.FC<ChatMessageProps> = ({ message }) => {
    const isUser = message.role === 'user'
    const messageText = useMemo(() => {
        if (typeof message?.text === 'string') return message.text
        if (message?.text == null) return ''
        try {
            return JSON.stringify(message.text)
        } catch {
            return String(message.text)
        }
    }, [message?.text])

    const { displayMarkdown, traceSteps, totalSeconds, showTrace, cached, streaming, partialTotal } = useMemo(() => {
        if (isUser) {
            return {
                displayMarkdown: messageText,
                traceSteps: [] as ReasoningStep[],
                showTrace: false,
                streaming: false,
                partialTotal: undefined as number | undefined,
            }
        }
        const st = message.meta?.streaming === true
        const fromMeta = message.meta?.reasoning_trace
        // 流式占位：有 streaming 标记即展示思考区（trace 可能仍为空数组）
        if (st || (fromMeta != null && fromMeta.length > 0)) {
            return {
                displayMarkdown: messageText,
                traceSteps: fromMeta || [],
                totalSeconds: message.meta?.total_seconds,
                showTrace: true,
                cached: message.meta?.cached,
                streaming: st,
                partialTotal: message.meta?.partial_total_seconds,
            }
        }
        const legacy = parseLegacyQueryBlock(messageText)
        if (legacy && legacy.steps.length > 0) {
            return {
                displayMarkdown: legacy.bodyMarkdown,
                traceSteps: legacy.steps,
                totalSeconds: legacy.totalSeconds,
                showTrace: true,
                cached: false,
                streaming: false,
                partialTotal: undefined,
            }
        }
        return {
            displayMarkdown: messageText,
            traceSteps: [] as ReasoningStep[],
            showTrace: false,
            streaming: false,
            partialTotal: undefined,
        }
    }, [isUser, messageText, message.meta])

    return (
        <div className="flex justify-start px-4 py-2">
            <div
                className={`max-w-[70%] rounded-3xl px-4 py-3 shadow-sm transition-transform hover:-translate-y-1 ${
                    isUser
                        ? 'bg-blue-500 text-white rounded-tr-none'
                        : `bg-white bg-glass-light backdrop-blur-[10px] border border-glass-border rounded-tl-none ${
                              !messageText.trim() && (streaming || showTrace) ? 'min-h-[4.5rem]' : ''
                          }`
                }`}
            >
                <div className={isUser ? 'text-white' : 'text-gray-800'}>
                    {isUser ? (
                        <p className="text-sm">{messageText}</p>
                    ) : (
                        <>
                            {/* 不用 prose 包裹 details，避免空内容/折叠区被 Typography 吃掉 */}
                            {showTrace ? (
                                <div className="text-sm not-prose mb-2">
                                    <ReasoningDetails
                                        steps={traceSteps}
                                        totalSeconds={totalSeconds}
                                        cached={cached}
                                        streaming={streaming}
                                        partialTotal={partialTotal}
                                    />
                                </div>
                            ) : null}
                            <div
                                className={
                                    displayMarkdown.trim() || streaming
                                        ? 'text-sm prose prose-sm dark:prose-invert max-w-none'
                                        : 'text-sm'
                                }
                            >
                                {displayMarkdown.trim() ? (
                                    <ReactMarkdown
                                        components={{
                                            p: ({ node, ...props }) => <p className="mb-2 last:mb-0" {...props} />,
                                            ul: ({ node, ...props }) => (
                                                <ul className="list-disc list-inside mb-2" {...props} />
                                            ),
                                            ol: ({ node, ...props }) => (
                                                <ol className="list-decimal list-inside mb-2" {...props} />
                                            ),
                                            li: ({ node, ...props }) => <li className="mb-1" {...props} />,
                                            code: (props) => {
                                                const { children, className } = props
                                                const inline = !className?.includes('language-')
                                                return inline ? (
                                                    <code className="bg-gray-200 px-1 rounded text-sm">{children}</code>
                                                ) : (
                                                    <code className="block bg-gray-900 text-white p-2 rounded mb-2 overflow-x-auto">
                                                        {children}
                                                    </code>
                                                )
                                            },
                                            strong: ({ node, ...props }) => <strong className="font-bold" {...props} />,
                                        }}
                                    >
                                        {displayMarkdown}
                                    </ReactMarkdown>
                                ) : streaming ? (
                                    <p className="text-xs text-slate-500 flex items-center gap-2 not-prose">
                                        <span className="inline-block h-1.5 w-1.5 rounded-full bg-violet-500 animate-pulse" />
                                        正在生成回答…
                                    </p>
                                ) : !showTrace && !messageText.trim() ? (
                                    <p className="text-xs text-slate-400 not-prose">等待回复…</p>
                                ) : null}
                            </div>
                        </>
                    )}
                </div>
            </div>
        </div>
    )
}
