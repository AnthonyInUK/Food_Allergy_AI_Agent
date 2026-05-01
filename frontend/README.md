# Food Allergy AI Agent - Next.js Frontend

A modern Next.js frontend for the Food Allergy AI Agent application.

## Features

- 🥗 Chat interface for asking about food allergens
- 📸 Upload food images for analysis
- 💬 Conversation management (create, select, delete)
- 🎨 Modern UI with glassmorphism design
- ⚡ Real-time message streaming
- 📱 Responsive design

## Installation

```bash
cd frontend
npm install
```

## Environment Setup

Create a `.env.local` file in the `frontend` directory:

```bash
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Running the Development Server

```bash
npm run dev
```

Open [http://localhost:3000](http://localhost:3000) with your browser to see the result.

## Project Structure

```
src/
├── app/              # Next.js app directory
│   ├── page.tsx      # Main chat page
│   ├── layout.tsx    # Root layout
│   └── globals.css   # Global styles
├── components/       # React components
│   ├── Sidebar.tsx        # Conversation sidebar
│   ├── ChatHistory.tsx     # Message history display
│   ├── ChatMessage.tsx     # Individual message component
│   └── ChatInput.tsx       # Message input area
├── context/          # React contexts
│   └── ConversationContext.tsx  # Global conversation state
└── lib/              # Utilities and helpers
    └── api.ts        # API client
```

## Features

### Conversation Management
- Create new conversations
- Switch between conversations
- Delete conversations
- Auto-save conversation history

### Chat Interface
- Send text messages
- Upload and analyze food images
- Real-time response streaming
- Markdown support for responses
- Typing indicators

### UI/UX
- Glassmorphism design
- Smooth animations
- Dark mode ready
- Mobile responsive
- Accessible components

## API Integration

The frontend connects to a FastAPI backend. Ensure the backend is running before starting the frontend.

Backend endpoints:
- `GET /api/conversations` - List all conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations/{id}` - Get conversation details
- `DELETE /api/conversations/{id}` - Delete conversation
- `POST /api/chat` - Send message
- `POST /api/upload` - Upload image

## Building for Production

```bash
npm run build
npm start
```

## Technologies Used

- **Next.js 14** - React framework
- **TypeScript** - Type safety
- **Tailwind CSS** - Styling
- **Axios** - HTTP client
- **React Markdown** - Markdown rendering
- **Lucide React** - Icons
- **date-fns** - Date formatting

## License

MIT
