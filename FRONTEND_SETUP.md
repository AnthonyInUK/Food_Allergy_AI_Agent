# Quick Start Guide - Food Allergy AI Agent Frontend

## Prerequisites

- Node.js 18+ and npm/yarn
- The FastAPI backend running on `http://localhost:8000`

## Installation & Setup

### Option 1: Manual Setup

```bash
cd frontend
npm install
cp .env.local.example .env.local
npm run dev
```

### Option 2: Using Setup Script

```bash
cd frontend
chmod +x setup.sh
./setup.sh
npm run dev
```

### Option 3: Docker Compose

```bash
# From project root
docker-compose up --build
```

The frontend will be available at `http://localhost:3000`

## Development

### Available Scripts

```bash
# Start development server with hot reload
npm run dev

# Build for production
npm run build

# Start production server
npm start

# Run ESLint
npm run lint
```

## Environment Variables

Create a `.env.local` file in the `frontend` directory:

```bash
# API Server URL (adjust if your backend is on a different host)
NEXT_PUBLIC_API_URL=http://localhost:8000
```

## Features

### Chat Interface
- Send text messages to the AI agent
- Get allergen analysis and recommendations

### Image Upload
- Click the **+** button to upload food images
- The AI will analyze the image for allergens

### Conversation Management
- **New Conversation**: Start a fresh chat
- **Select Conversation**: Switch between past conversations
- **Delete Conversation**: Remove a conversation

## Project Structure

```
frontend/
├── src/
│   ├── app/                 # Next.js pages and layouts
│   ├── components/          # React components
│   │   ├── ChatInput.tsx         # Message input with file upload
│   │   ├── ChatMessage.tsx       # Individual message display
│   │   ├── ChatHistory.tsx       # Message history
│   │   └── Sidebar.tsx           # Conversation sidebar
│   ├── context/             # React Context providers
│   │   └── ConversationContext.tsx  # Global state management
│   └── lib/
│       └── api.ts           # API client and types
├── package.json
├── tsconfig.json
├── tailwind.config.ts
├── next.config.js
└── README.md
```

## API Endpoints

The frontend communicates with these backend endpoints:

- `GET /api/conversations` - List all conversations
- `POST /api/conversations` - Create new conversation
- `GET /api/conversations/{id}` - Get conversation details
- `PUT /api/conversations/{id}` - Update conversation
- `DELETE /api/conversations/{id}` - Delete conversation
- `POST /api/chat` - Send message and get AI response
- `POST /api/upload` - Upload image for analysis

## Styling

The frontend uses:
- **Tailwind CSS** for utility-first styling
- **Glassmorphism** design with backdrop blur and transparency
- **Lucide React** for icons
- Custom animations and transitions

## Troubleshooting

### Frontend won't connect to backend
- Ensure the API server is running on `http://localhost:8000`
- Check `NEXT_PUBLIC_API_URL` in `.env.local`
- Check browser console for CORS errors

### Conversations not loading
- Verify the backend API is running
- Check if `conversations.json` exists in the backend's `data/` directory

### Styling issues
- Run `npm run build` to rebuild Tailwind CSS
- Clear `.next` folder and restart dev server

## Learn More

- [Next.js Documentation](https://nextjs.org/docs)
- [Tailwind CSS Documentation](https://tailwindcss.com/docs)
- [React Documentation](https://react.dev)

## License

MIT
