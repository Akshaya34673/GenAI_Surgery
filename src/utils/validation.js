// src/utils/validation.js

/*
 * Validates email format with strict real-world rules
 */
export const isValidEmail = (email) => {
  if (!email || typeof email !== 'string') {
    return false;
  }

  email = email.trim();

  if (email.length > 254) {
    return false;
  }

  if (email.includes('..')) {
    return false;
  }

  if ((email.match(/@/g) || []).length !== 1) {
    return false;
  }

  try {
    const [localPart, domain] = email.split('@');

    // Local Part Validation
    if (!localPart || localPart.length < 1 || localPart.length > 64) {
      return false;
    }

    if (localPart.startsWith('.') || localPart.endsWith('.')) {
      return false;
    }

    // Must contain at least one letter
    if (!/[a-zA-Z]/.test(localPart)) {
      return false;
    }

    // Must start and end with alphanumeric
    if (!/^[a-zA-Z0-9]/.test(localPart) || !/[a-zA-Z0-9]$/.test(localPart)) {
      return false;
    }

    if (!/^[a-zA-Z0-9._%+-]+$/.test(localPart)) {
      return false;
    }

    // Minimum 3 characters for realistic emails
    if (localPart.length < 3) {
      return false;
    }

    // Domain Validation
    if (!domain || domain.length < 4) {
      return false;
    }

    if (domain.startsWith('.') || domain.endsWith('.') ||
        domain.startsWith('-') || domain.endsWith('-')) {
      return false;
    }

    if (!domain.includes('.')) {
      return false;
    }

    const domainParts = domain.split('.');

    if (domainParts.length < 2) {
      return false;
    }

    const tld = domainParts[domainParts.length - 1];
    if (tld.length < 2 || !/^[a-zA-Z]+$/.test(tld)) {
      return false;
    }

    const domainName = domainParts[domainParts.length - 2];
    if (domainName.length < 2) {
      return false;
    }

    for (const part of domainParts) {
      if (!part) {
        return false;
      }

      if (!/^[a-zA-Z0-9-]+$/.test(part)) {
        return false;
      }

      if (part.startsWith('-') || part.endsWith('-')) {
        return false;
      }

      if (!/[a-zA-Z]/.test(part)) {
        return false;
      }
    }

    return true;

  } catch (error) {
    return false;
  }
};

/**
 * Get user-friendly error message for invalid email
 */
export const getEmailError = (email) => {
  if (!email) {
    return 'Email is required';
  }

  email = email.trim();

  if (email.includes('..')) {
    return 'Email cannot contain consecutive dots';
  }

  if ((email.match(/@/g) || []).length !== 1) {
    return 'Email must contain exactly one @ symbol';
  }

  const [localPart, domain] = email.split('@');

  if (!localPart) {
    return 'Email must have a username before @';
  }

  if (localPart.startsWith('.') || localPart.endsWith('.')) {
    return 'Username cannot start or end with a dot';
  }

  if (!/[a-zA-Z]/.test(localPart)) {
    return 'Username must contain at least one letter';
  }

  if (!/^[a-zA-Z0-9]/.test(localPart)) {
    return 'Username must start with a letter or number';
  }

  if (localPart.length < 3) {
    return 'Username must be at least 3 characters';
  }

  if (!domain) {
    return 'Email must have a domain after @';
  }

  if (!domain.includes('.')) {
    return 'Domain must include a dot (e.g., gmail.com)';
  }

  const domainParts = domain.split('.');
  const tld = domainParts[domainParts.length - 1];

  if (tld.length < 2) {
    return 'Domain extension must be at least 2 characters';
  }

  return 'Please enter a valid email address';
};