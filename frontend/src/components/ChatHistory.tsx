'use client'

import React, { useEffect, useRef } from 'react'
import { ChatMessage } from './ChatMessage'
import { useConversation } from '@/context/ConversationContext'

export const ChatHistory: React.FC = () => {
    const { currentConversation, isLoading } = useConversation()
    const messagesEndRef = useRef<HTMLDivElement>(null)

    const scrollToBottom = () => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }

    useEffect(() => {
        scrollToBottom()
    }, [currentConversation?.messages])

    if (!currentConversation) {
        return (
            <div className="flex-1 min-h-0 flex items-center justify-center text-gray-400">
                <p>Loading conversation...</p>
            </div>
        )
    }

    return (
        <div className="flex-1 min-h-0 min-w-0 overflow-y-auto flex flex-col">
            {currentConversation.messages.length === 0 ? (
                <div className="flex-1 flex items-center justify-center text-center text-gray-400 px-4">
                    <div>
                        <p className="text-2xl mb-2">🥗</p>
                        <p className="text-lg font-medium">Start a conversation</p>
                        <p className="text-sm mt-2">Upload food images or ask about allergies</p>
                    </div>
                </div>
            ) : (
                <>
                    {currentConversation.messages.map((message, idx) => (
                        <ChatMessage key={idx} message={message} />
                    ))}
                    {isLoading && (
                        <div className="flex justify-start px-4 py-2">
                            <div className="flex gap-2 items-center bg-white bg-glass-light backdrop-blur-[10px] border border-glass-border rounded-3xl rounded-tl-none px-4 py-3">
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce" />
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-100" />
                                <div className="w-2 h-2 bg-gray-400 rounded-full animate-bounce delay-200" />
                            </div>
                        </div>
                    )}
                    <div ref={messagesEndRef} />
                </>
            )}
        </div>
    )
}
