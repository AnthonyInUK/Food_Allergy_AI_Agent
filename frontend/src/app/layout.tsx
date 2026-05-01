import type { Metadata } from 'next'
import './globals.css'
import { ConversationProvider } from '@/context/ConversationContext'

export const metadata: Metadata = {
    title: 'Food Allergy AI Agent',
    description: 'Upload food images or ask questions to get allergen information',
}

export default function RootLayout({
    children,
}: {
    children: React.ReactNode
}) {
    return (
        <html lang="en">
            <body>
                <ConversationProvider>{children}</ConversationProvider>
            </body>
        </html>
    )
}
