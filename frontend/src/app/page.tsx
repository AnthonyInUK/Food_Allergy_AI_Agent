'use client'

import React from 'react'
import { Sidebar } from '@/components/Sidebar'
import { ChatHistory } from '@/components/ChatHistory'
import { ChatInput } from '@/components/ChatInput'

export default function Home() {
  return (
    <div className="w-screen h-screen bg-gradient overflow-hidden flex">
      {/* Sidebar */}
      <Sidebar />

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col min-h-0 min-w-0">
        {/* Header */}
        <div className="px-6 py-4 border-b border-gray-200 shadow-sm">
          <h1 className="text-2xl font-bold text-gray-800 flex items-center gap-2">
            🥗 <span>Food Allergy AI Agent</span>
          </h1>
          <p className="text-sm text-gray-600 mt-1">
            Upload food images or ask questions to check for allergens
          </p>
        </div>

        {/* Chat Messages */}
        <ChatHistory />

        {/* Chat Input */}
        <ChatInput />
      </div>
    </div>
  )
}
