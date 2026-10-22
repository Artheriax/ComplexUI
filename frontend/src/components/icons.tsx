interface IconProps {
  className?: string;
}

export function PencilIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M13.586 3.586a2 2 0 1 1 2.828 2.828l-8.5 8.5a1 1 0 0 1-.44.256l-3.2.914a.5.5 0 0 1-.62-.62l.914-3.2a1 1 0 0 1 .256-.44l8.762-8.238Z" />
    </svg>
  );
}

export function TrashIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M8.75 1A2.75 2.75 0 0 0 6 3.75v.443c-.795.077-1.584.176-2.365.298a.75.75 0 1 0 .23 1.482l.149-.022.841 10.518A2.75 2.75 0 0 0 7.596 19h4.807a2.75 2.75 0 0 0 2.742-2.53l.841-10.52.149.023a.75.75 0 0 0 .23-1.482A41.03 41.03 0 0 0 14 4.193V3.75A2.75 2.75 0 0 0 11.25 1h-2.5ZM10 4c.84 0 1.673.025 2.5.075V3.75c0-.69-.56-1.25-1.25-1.25h-2.5c-.69 0-1.25.56-1.25 1.25v.325C8.327 4.025 9.16 4 10 4ZM8.58 7.72a.75.75 0 0 0-1.5.06l.3 7.5a.75.75 0 1 0 1.5-.06l-.3-7.5Zm4.34.06a.75.75 0 1 0-1.5-.06l-.3 7.5a.75.75 0 1 0 1.5.06l.3-7.5Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function WarningIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M8.485 2.495c.673-1.167 2.357-1.167 3.03 0l6.28 10.875c.673 1.167-.17 2.625-1.516 2.625H3.72c-1.347 0-2.189-1.458-1.515-2.625L8.485 2.495ZM10 5a.75.75 0 0 1 .75.75v3.5a.75.75 0 0 1-1.5 0v-3.5A.75.75 0 0 1 10 5Zm0 9a1 1 0 1 0 0-2 1 1 0 0 0 0 2Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function PlayIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M6.3 2.84A1.5 1.5 0 0 0 4 4.11v11.78a1.5 1.5 0 0 0 2.3 1.27l9.344-5.891a1.5 1.5 0 0 0 0-2.538L6.3 2.841Z" />
    </svg>
  );
}

export function SpinnerIcon({ className = "h-3.5 w-3.5 animate-spin" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 24 24" fill="none" aria-hidden="true">
      <circle cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4" className="opacity-25" />
      <path d="M12 2a10 10 0 0 1 10 10" stroke="currentColor" strokeWidth="4" strokeLinecap="round" />
    </svg>
  );
}

export function CloseIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M6.28 5.22a.75.75 0 0 0-1.06 1.06L8.94 10l-3.72 3.72a.75.75 0 1 0 1.06 1.06L10 11.06l3.72 3.72a.75.75 0 1 0 1.06-1.06L11.06 10l3.72-3.72a.75.75 0 0 0-1.06-1.06L10 8.94 6.28 5.22Z" />
    </svg>
  );
}

export function GripIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 16 16" fill="currentColor" aria-hidden="true">
      <circle cx="5" cy="3" r="1.2" />
      <circle cx="11" cy="3" r="1.2" />
      <circle cx="5" cy="8" r="1.2" />
      <circle cx="11" cy="8" r="1.2" />
      <circle cx="5" cy="13" r="1.2" />
      <circle cx="11" cy="13" r="1.2" />
    </svg>
  );
}

export function ResetIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M15.312 11.424a5.5 5.5 0 0 1-9.201 2.466l-.312-.311h2.433a.75.75 0 0 0 0-1.5H3.989a.75.75 0 0 0-.75.75v4.242a.75.75 0 0 0 1.5 0v-2.43l.31.31a7 7 0 0 0 11.712-3.138.75.75 0 0 0-1.449-.39Zm1.23-3.723a.75.75 0 0 0 .219-.53V2.929a.75.75 0 0 0-1.5 0V5.36l-.31-.31A7 7 0 0 0 3.239 8.188a.75.75 0 1 0 1.448.389A5.5 5.5 0 0 1 13.89 6.11l.311.31h-2.432a.75.75 0 0 0 0 1.5h4.243a.75.75 0 0 0 .53-.219Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function CheckIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M16.704 4.153a.75.75 0 0 1 .143 1.052l-8 10.5a.75.75 0 0 1-1.127.075l-4.5-4.5a.75.75 0 0 1 1.06-1.06l3.894 3.893 7.48-9.817a.75.75 0 0 1 1.05-.143Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function BoltIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M14.615 1.595a.75.75 0 0 1 .359.852L12.982 9.75h7.268a.75.75 0 0 1 .548 1.262l-10.5 11.25a.75.75 0 0 1-1.272-.71l1.992-7.302H3.75a.75.75 0 0 1-.548-1.262l10.5-11.25a.75.75 0 0 1 .913-.143Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function DuplicateIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M8 8a2 2 0 0 0-2 2v6a2 2 0 0 0 2 2h6a2 2 0 0 0 2-2v-6a2 2 0 0 0-2-2H8Z" />
      <path d="M6 2a2 2 0 0 0-2 2v8a2 2 0 0 0 2 2h1v-2H6V4h8v1h2V4a2 2 0 0 0-2-2H6Z" />
    </svg>
  );
}

export function SaveIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M4.75 2A2.75 2.75 0 0 0 2 4.75v10.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25V6.414a2.75 2.75 0 0 0-.805-1.944l-1.665-1.665A2.75 2.75 0 0 0 13.586 2H4.75Zm0 1.5h2.5v4.25c0 .69.56 1.25 1.25 1.25h4.5c.69 0 1.25-.56 1.25-1.25V3.75h.336c.332 0 .65.132.884.366l1.665 1.665c.234.234.365.552.365.884V15.25c0 .69-.56 1.25-1.25 1.25h-.5v-5.25c0-.69-.56-1.25-1.25-1.25h-6.5c-.69 0-1.25.56-1.25 1.25V16.5h-.5c-.69 0-1.25-.56-1.25-1.25V4.75c0-.69.56-1.25 1.25-1.25ZM8.75 11h4.5v5.5h-4.5V11Z" />
    </svg>
  );
}

export function DownloadIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M10.75 2.75a.75.75 0 0 0-1.5 0v8.614L6.295 8.235a.75.75 0 1 0-1.09 1.03l4.25 4.5a.75.75 0 0 0 1.09 0l4.25-4.5a.75.75 0 0 0-1.09-1.03l-2.955 3.129V2.75Z" />
      <path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" />
    </svg>
  );
}

export function UploadIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M9.25 13.25a.75.75 0 0 0 1.5 0V4.636l2.955 3.129a.75.75 0 0 0 1.09-1.03l-4.25-4.5a.75.75 0 0 0-1.09 0l-4.25 4.5a.75.75 0 1 0 1.09 1.03L9.25 4.636v8.614Z" />
      <path d="M3.5 12.75a.75.75 0 0 0-1.5 0v2.5A2.75 2.75 0 0 0 4.75 18h10.5A2.75 2.75 0 0 0 18 15.25v-2.5a.75.75 0 0 0-1.5 0v2.5c0 .69-.56 1.25-1.25 1.25H4.75c-.69 0-1.25-.56-1.25-1.25v-2.5Z" />
    </svg>
  );
}

export function PlusIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M10.75 4.75a.75.75 0 0 0-1.5 0v4.5h-4.5a.75.75 0 0 0 0 1.5h4.5v4.5a.75.75 0 0 0 1.5 0v-4.5h4.5a.75.75 0 0 0 0-1.5h-4.5v-4.5Z" />
    </svg>
  );
}

export function StopIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <rect x="5" y="5" width="10" height="10" rx="1.5" />
    </svg>
  );
}

export function ChevronDownIcon({ className = "h-3 w-3" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M5.22 7.22a.75.75 0 0 1 1.06 0L10 10.94l3.72-3.72a.75.75 0 1 1 1.06 1.06l-4.25 4.25a.75.75 0 0 1-1.06 0L5.22 8.28a.75.75 0 0 1 0-1.06Z" />
    </svg>
  );
}

export function UndoIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M7.793 2.232a.75.75 0 0 1-.025 1.06L3.622 7.25h10.003a5.375 5.375 0 0 1 0 10.75H10.75a.75.75 0 0 1 0-1.5h2.875a3.875 3.875 0 0 0 0-7.75H3.622l4.146 3.957a.75.75 0 0 1-1.036 1.085l-5.5-5.25a.75.75 0 0 1 0-1.085l5.5-5.25a.75.75 0 0 1 1.061.025Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function RedoIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M12.207 2.232a.75.75 0 0 0 .025 1.06l4.146 3.958H6.375a5.375 5.375 0 0 0 0 10.75H9.25a.75.75 0 0 0 0-1.5H6.375a3.875 3.875 0 0 1 0-7.75h10.003l-4.146 3.957a.75.75 0 0 0 1.036 1.085l5.5-5.25a.75.75 0 0 0 0-1.085l-5.5-5.25a.75.75 0 0 0-1.06.025Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function SettingsIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M7.84 1.804A1 1 0 0 1 8.82 1h2.36a1 1 0 0 1 .98.804l.331 1.652a6.993 6.993 0 0 1 1.929 1.115l1.598-.54a1 1 0 0 1 1.186.447l1.18 2.044a1 1 0 0 1-.205 1.251l-1.267 1.113a7.047 7.047 0 0 1 0 2.228l1.267 1.113a1 1 0 0 1 .206 1.25l-1.18 2.045a1 1 0 0 1-1.187.447l-1.598-.54a6.993 6.993 0 0 1-1.929 1.115l-.33 1.652a1 1 0 0 1-.98.804H8.82a1 1 0 0 1-.98-.804l-.331-1.652a6.993 6.993 0 0 1-1.929-1.115l-1.598.54a1 1 0 0 1-1.186-.447l-1.18-2.044a1 1 0 0 1 .205-1.251l1.267-1.114a7.05 7.05 0 0 1 0-2.227L1.821 7.773a1 1 0 0 1-.206-1.25l1.18-2.045a1 1 0 0 1 1.187-.447l1.598.54A6.992 6.992 0 0 1 7.51 3.456l.33-1.652ZM10 13a3 3 0 1 0 0-6 3 3 0 0 0 0 6Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function FileIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M4.5 2A1.5 1.5 0 0 0 3 3.5v13A1.5 1.5 0 0 0 4.5 18h11a1.5 1.5 0 0 0 1.5-1.5V7.621a1.5 1.5 0 0 0-.44-1.06l-4.12-4.122A1.5 1.5 0 0 0 11.378 2H4.5Zm2.25 8.25a.75.75 0 0 1 .75-.75h5.5a.75.75 0 0 1 0 1.5H7.5a.75.75 0 0 1-.75-.75Zm0 3a.75.75 0 0 1 .75-.75h3.5a.75.75 0 0 1 0 1.5H7.5a.75.75 0 0 1-.75-.75Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function EyeIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M10 12.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" />
      <path
        fillRule="evenodd"
        d="M.664 10.59a1.651 1.651 0 0 1 0-1.186A10.004 10.004 0 0 1 10 3c4.257 0 7.893 2.66 9.336 6.41.147.381.146.804 0 1.186A10.004 10.004 0 0 1 10 17c-4.257 0-7.893-2.66-9.336-6.41ZM14 10a4 4 0 1 1-8 0 4 4 0 0 1 8 0Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function SearchIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M9 3.5a5.5 5.5 0 1 0 3.474 9.762l3.132 3.132a.75.75 0 1 0 1.06-1.06l-3.132-3.132A5.5 5.5 0 0 0 9 3.5ZM5 9a4 4 0 1 1 8 0 4 4 0 0 1-8 0Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function CopyIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M7 3.5A1.5 1.5 0 0 1 8.5 2h6A1.5 1.5 0 0 1 16 3.5v9a1.5 1.5 0 0 1-1.5 1.5h-6A1.5 1.5 0 0 1 7 12.5v-9Z" />
      <path d="M4 6.5A1.5 1.5 0 0 1 5.5 5H6v7.5A2.5 2.5 0 0 0 8.5 15H12v.5a1.5 1.5 0 0 1-1.5 1.5h-5A1.5 1.5 0 0 1 4 15.5v-9Z" />
    </svg>
  );
}

export function CutIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M7.5 2.75a.75.75 0 0 0-1.5 0v6.09a3.5 3.5 0 1 0 1.5 2.872V9.25l7.72 5.145a.75.75 0 0 0 .83-1.25L8.55 8 16.05 3a.75.75 0 1 0-.83-1.25L7.5 6.61V2.75Zm-2 11.5a2 2 0 1 1 4 0 2 2 0 0 1-4 0Z" />
      <path d="M13 7.5a2.5 2.5 0 1 0 0-5 2.5 2.5 0 0 0 0 5Z" opacity=".65" />
    </svg>
  );
}

export function PasteIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M7 3a2 2 0 0 1 2-2h2a2 2 0 0 1 2 2h1.5A1.5 1.5 0 0 1 16 4.5v12a1.5 1.5 0 0 1-1.5 1.5h-9A1.5 1.5 0 0 1 4 16.5v-12A1.5 1.5 0 0 1 5.5 3H7Zm1.5-1a.5.5 0 0 0-.5.5V4h4v-1.5a.5.5 0 0 0-.5-.5h-3Z" />
    </svg>
  );
}

export function EyeOffIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M3.28 2.22a.75.75 0 0 0-1.06 1.06l14.5 14.5a.75.75 0 1 0 1.06-1.06l-1.745-1.745a10.029 10.029 0 0 0 3.3-4.38 1.651 1.651 0 0 0 0-1.185A10.004 10.004 0 0 0 9.999 3a9.956 9.956 0 0 0-4.744 1.194L3.28 2.22ZM7.752 6.69l1.092 1.092a2.5 2.5 0 0 1 3.374 3.373l1.091 1.092a4 4 0 0 0-5.557-5.557Z"
        clipRule="evenodd"
      />
      <path d="m10.748 13.93 2.523 2.523a9.987 9.987 0 0 1-3.27.547c-4.258 0-7.894-2.66-9.337-6.41a1.651 1.651 0 0 1 0-1.186A10.007 10.007 0 0 1 2.839 6.02L6.07 9.252a4 4 0 0 0 4.678 4.678Z" />
    </svg>
  );
}

export type ArrangeIconKind =
  | "align-left"
  | "align-center-x"
  | "align-right"
  | "align-top"
  | "align-center-y"
  | "align-bottom"
  | "distribute-x"
  | "distribute-y"
  | "grid";

const ARRANGE_PATHS: Record<ArrangeIconKind, string> = {
  "align-left":
    "M2.5 2.5h1.5v15H2.5zM6 4.5h9.5v4H6zM6 11.5h6.5v4H6z",
  "align-center-x":
    "M9.25 2.5h1.5v15h-1.5zM5 4.5h10v4H5zM6.5 11.5h7v4h-7z",
  "align-right":
    "M16 2.5h1.5v15H16zM4.5 4.5H14v4H4.5zM7.5 11.5H14v4H7.5z",
  "align-top":
    "M2.5 2.5h15v1.5h-15zM4.5 6h4v9.5h-4zM11.5 6h4v6.5h-4z",
  "align-center-y":
    "M2.5 9.25h15v1.5h-15zM4.5 5h4v10h-4zM11.5 6.5h4v7h-4z",
  "align-bottom":
    "M2.5 16h15v1.5h-15zM4.5 4.5h4V14h-4zM11.5 7.5h4V14h-4z",
  "distribute-x":
    "M2 5h2.5v10H2zM8.75 5h2.5v10h-2.5zM15.5 5H18v10h-2.5z",
  "distribute-y":
    "M5 2h10v2.5H5zM5 8.75h10v2.5H5zM5 15.5h10V18H5z",
  grid:
    "M2.5 2.5h6v6h-6zM11.5 2.5h6v6h-6zM2.5 11.5h6v6h-6zM11.5 11.5h6v6h-6z",
};

export function ArrangeIcon({
  kind,
  className = "h-3.5 w-3.5",
}: IconProps & { kind: ArrangeIconKind }) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d={ARRANGE_PATHS[kind]} />
    </svg>
  );
}

export function SubgraphIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M2.5 3h6v6h-6zM11.5 11h6v6h-6zM8.5 5h3.5v2H10.5v3h-2V5Z" />
      <path d="M11.5 3h6v6h-6z" opacity=".55" />
    </svg>
  );
}

export function EnterIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M3 3.5h8v2H5v9h6v2H3v-13Z" />
      <path d="M11.8 6.3 16.5 10l-4.7 3.7v-2.4H9v-2.6h2.8V6.3Z" />
    </svg>
  );
}

export function FocusIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M2.5 2.5h5v1.5H4v3.5H2.5v-5Zm10 0h5v5H16V4h-3.5V2.5Zm-10 10H4V16h3.5v1.5h-5v-5Zm13.5 0H17.5v5h-5V16H16v-3.5Z" />
    </svg>
  );
}

export function ArrowLeftIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="m9 4.2-5.8 5.8 5.8 5.8 1.4-1.4-3.4-3.4h9.5v-2H7l3.4-3.4-1.4-1.4Z" />
    </svg>
  );
}

export function HistoryIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M10 2.5a7.5 7.5 0 1 0 7.14 9.79l-1.43-.46A6 6 0 1 1 10 4v2.25l3-2.75-3-2.75V2.5Zm.9 4.5v4.2l3.1 1.85-.75 1.24-3.55-2.12V7h1.2Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function GroupIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M2.5 2.5h15v15h-15v-15Zm1.5 1.5v12h12V4H4Z"
        clipRule="evenodd"
      />
      <path d="M6 6h4v4H6zM10 10h4v4h-4z" />
    </svg>
  );
}

export function UngroupIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M2.5 2.5h7v7h-7zM10.5 10.5h7v7h-7zM11.5 2.5h6v6h-6z" opacity=".85" />
      <path d="M2.5 11.5h6v6h-6z" opacity=".55" />
    </svg>
  );
}

export function GridIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M2.5 2.5h6v6h-6zM11.5 2.5h6v6h-6zM2.5 11.5h6v6h-6zM11.5 11.5h6v6h-6z" />
    </svg>
  );
}

export function FolderIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M2 5.5A2.5 2.5 0 0 1 4.5 3h3.19a2 2 0 0 1 1.414.586L10.414 4.9H15.5A2.5 2.5 0 0 1 18 7.4v7.1a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 2 14.5v-9Z" />
    </svg>
  );
}

export function DatabaseIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M10 2c3.314 0 6 1.12 6 2.5S13.314 7 10 7 4 5.88 4 4.5 6.686 2 10 2Z" />
      <path d="M4 6.6c1.326.84 3.53 1.4 6 1.4s4.674-.56 6-1.4v2.2c0 1.38-2.686 2.5-6 2.5s-6-1.12-6-2.5V6.6Z" />
      <path d="M4 11.1c1.326.84 3.53 1.4 6 1.4s4.674-.56 6-1.4v2.2c0 1.38-2.686 2.5-6 2.5s-6-1.12-6-2.5v-2.2Z" />
      <path d="M4 15.6c1.326.84 3.53 1.4 6 1.4s4.674-.56 6-1.4V18c0 1.38-2.686 2.5-6 2.5S4 19.38 4 18v-2.4Z" opacity=".7" />
    </svg>
  );
}

export function TerminalIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M3.5 3A1.5 1.5 0 0 0 2 4.5v11A1.5 1.5 0 0 0 3.5 17h13a1.5 1.5 0 0 0 1.5-1.5v-11A1.5 1.5 0 0 0 16.5 3h-13Z" opacity=".25" />
      <path d="M5.22 6.72a.75.75 0 0 1 1.06 0l2.25 2.25a.75.75 0 0 1 0 1.06L6.28 12.28a.75.75 0 1 1-1.06-1.06L6.94 9.5 5.22 7.78a.75.75 0 0 1 0-1.06ZM9.5 12.5a.75.75 0 0 1 .75-.75h4a.75.75 0 0 1 0 1.5h-4a.75.75 0 0 1-.75-.75Z" />
    </svg>
  );
}

export function KeyboardIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path
        fillRule="evenodd"
        d="M2 6.5A2.5 2.5 0 0 1 4.5 4h11A2.5 2.5 0 0 1 18 6.5v7a2.5 2.5 0 0 1-2.5 2.5h-11A2.5 2.5 0 0 1 2 13.5v-7ZM4.5 5.5A1 1 0 0 0 3.5 6.5v7a1 1 0 0 0 1 1h11a1 1 0 0 0 1-1v-7a1 1 0 0 0-1-1h-11Zm.75 2a.75.75 0 0 1 .75.75v.5a.75.75 0 0 1-1.5 0v-.5A.75.75 0 0 1 5.25 7.5Zm3 0a.75.75 0 0 1 .75.75v.5a.75.75 0 0 1-1.5 0v-.5a.75.75 0 0 1 .75-.75Zm3 0a.75.75 0 0 1 .75.75v.5a.75.75 0 0 1-1.5 0v-.5a.75.75 0 0 1 .75-.75Zm3 0a.75.75 0 0 1 .75.75v.5a.75.75 0 0 1-1.5 0v-.5a.75.75 0 0 1 .75-.75ZM6 10.25a.75.75 0 0 0 0 1.5h8a.75.75 0 0 0 0-1.5H6Zm-.75 2.5a.75.75 0 0 0 0 1.5h9.5a.75.75 0 0 0 0-1.5h-9.5Z"
        clipRule="evenodd"
      />
    </svg>
  );
}

export function NewWindowIcon({ className = "h-3.5 w-3.5" }: IconProps) {
  return (
    <svg className={className} viewBox="0 0 20 20" fill="currentColor" aria-hidden="true">
      <path d="M3.5 3A1.5 1.5 0 0 0 2 4.5v11A1.5 1.5 0 0 0 3.5 17h13a1.5 1.5 0 0 0 1.5-1.5v-11A1.5 1.5 0 0 0 16.5 3h-13Z" opacity=".3" />
      <path d="M10.75 6.25a.75.75 0 0 0-1.5 0v3H6.25a.75.75 0 0 0 0 1.5h3v3a.75.75 0 0 0 1.5 0v-3h3a.75.75 0 0 0 0-1.5h-3v-3Z" />
    </svg>
  );
}
